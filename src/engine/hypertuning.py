"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import logging
import os
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

import src.constants as C

from typing import Any
from sklearn.model_selection import BaseCrossValidator
from optuna.study import Study
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration.xgboost import XGBoostPruningCallback

from .callbacks import BoosterCollector
from .cv import TimeSeriesCV
from .metrics import compute_metrics


logger = logging.getLogger(__name__)


def _save_trial_artifacts(
    study_name: str,
    trial: Trial,
    history: pd.DataFrame,
    boosters: list[xgb.Booster] | None,
) -> None:
    """Persist CV history and fold boosters to disk, keyed by study/trial.

    Args:
        study_name: Name of the Optuna study for directory organization.
        trial: Optuna trial object to store artifact metadata.
        history: DataFrame containing cross-validation history from xgb.cv().
        boosters: List of trained XGBoost booster objects (one per fold), or None.
    """
    trial_dir = os.path.join(C.LOG_DIR, study_name, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save full CV history
    history_path = os.path.join(trial_dir, "cv_history.csv")
    history.to_csv(history_path, index=True)
    trial.set_user_attr("artifact_dir", trial_dir)
    trial.set_user_attr("cv_history_path", history_path)

    # Save fold boosters
    if boosters:
        booster_paths = []
        for i, bst in enumerate(boosters):
            bst_path = os.path.join(trial_dir, f"fold_{i}.ubj")
            bst.save_model(bst_path)
            booster_paths.append(bst_path)
        trial.set_user_attr("booster_paths", booster_paths)

    logger.debug(f"Trial {trial.number} artifacts saved to {trial_dir}")


def _log_study(study: Study) -> None:
    """Persist summary statistics for a completed Optuna study to the storage DB.

    Stores trial counts, duration statistics, objective value statistics, and
    the best trial's number and parameters as a single ``"summary"`` dict on
    the study via ``study.set_user_attr``.  Best-trial params and per-trial
    user attributes are already persisted individually by Optuna; this adds a
    study-level roll-up that can be queried without loading every trial.

    Args:
        study: A completed (or partial) Optuna study.
    """
    trials = study.trials
    n_total   = len(trials)
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    n_pruned   = sum(t.state == optuna.trial.TrialState.PRUNED   for t in trials)
    n_failed   = sum(t.state == optuna.trial.TrialState.FAIL     for t in trials)
    p_pruned   = 100 * n_pruned / n_total if n_total > 0 else 0.0

    durations       = [t.duration.total_seconds() for t in trials if t.duration is not None]
    complete_values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

    summary: dict = {
        "n_total":    n_total,
        "n_complete": n_complete,
        "n_pruned":   n_pruned,
        "pct_pruned": round(p_pruned, 1),
        "n_failed":   n_failed,
    }

    if durations:
        summary["duration_total_s"] = round(sum(durations), 1)
        summary["duration_mean_s"]  = round(float(np.mean(durations)), 2)
        summary["duration_max_s"]   = round(float(max(durations)), 2)

    if complete_values:
        summary["best_value"]   = round(float(min(complete_values)), 6)
        summary["worst_value"]  = round(float(max(complete_values)), 6)
        summary["median_value"] = round(float(np.median(complete_values)), 6)

    best_trial = study.best_trial
    summary["best_trial_number"] = best_trial.number
    summary["best_trial_params"] = best_trial.params  # already a dict of JSON-serializable primitives

    study.set_user_attr("summary", summary)
    logger.info("Study '%s' summary persisted to Optuna DB.", study.study_name)

    return


def _log_trial_cv_results(
    trial: Trial,
    metric: str,
    history: pd.DataFrame,
) -> None:
    """Log CV results and store them as trial user attributes.

    Extracts best and final round metrics from CV history and stores them
    as trial user attributes for later analysis and reporting.

    Args:
        trial: Optuna trial object.
        metric: Metric name (e.g., "mae") used in XGBoost evaluation.
        history: DataFrame containing CV results for each boosting round,
            with columns like 'test-{metric}-mean' and 'test-{metric}-std'.
    """
    test_mean_col = f"test-{metric}-mean"
    test_std_col  = f"test-{metric}-std"
    final_loss      = history[test_mean_col].values[-1]
    final_loss_std  = history[test_std_col].values[-1] if test_std_col in history.columns else float("nan")
    best_loss       = history[test_mean_col].min()
    best_n_rounds = int(history[test_mean_col].idxmin()) + 1  # 1-based

    trial.set_user_attr("best_n_rounds",      best_n_rounds)
    trial.set_user_attr("final_loss_mean",    float(final_loss))
    trial.set_user_attr("final_loss_std",     float(final_loss_std))
    trial.set_user_attr("best_loss_mean",     float(best_loss))

    logger.debug(
        f"Trial {trial.number} | CV finished — "
        f"rounds used: {best_n_rounds}, "
        f"final {metric}: {final_loss:.4f} ± {final_loss_std:.4f}, "
        f"best {metric}: {best_loss:.4f} at round {best_n_rounds}"
    )


def _suggest_params(trial: Trial) -> dict[str, Any]:
    """Suggest hyperparameters for a trial using Optuna's sampling methods.

    Dynamically builds a hyperparameter dictionary based on the HYPERPARAMETERS
    configuration, using the appropriate Optuna suggest method for each parameter.

    Args:
        trial: Active Optuna trial.

    Returns:
        Dictionary mapping hyperparameter names to suggested values.
    """
    params = {}
    for name, (method, low, high, kwargs) in C.HYPERPARAMETERS.items():
        suggest_fn = getattr(trial, method)
        params[name] = suggest_fn(name, low, high, **kwargs)
    return params


def _objective(
    trial: Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    study_name: str = "",
    metric: str = C.XGB_CONSTANTS["eval_metric"],
    early_stopping_rounds: int = C.EARLY_STOPPING_ROUNDS,
    boosting_rounds: int = C.NUM_BOOST_ROUNDS,
    monitor_periods: int = C.MONITOR_PERIODS,
    xgboost_constants: dict = C.XGB_CONSTANTS,
) -> float:
    """Optuna objective for broad search over the full hyperparameter space.

    Suggests values from wide ranges for all XGBoost hyperparameters, performs
    nested time-series cross-validation, and records per-fold models and metrics.
    Integrates per-fold pruning via XGBoostPruningCallback for early termination
    of unpromising trials.

    Args:
        trial: Active Optuna trial.
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        cv: Time-series cross-validation splitter.
        study_name: Name of the Optuna study, used for organizing artifacts.
        metric: Evaluation metric name to monitor (e.g., "mae").
        early_stopping_rounds: Number of rounds with no improvement to trigger early stopping.
        boosting_rounds: Maximum number of boosting rounds to train.
        monitor_periods: Number of CV rounds to report in pruning callback (must be <= LOG_PERIOD).
        xgboost_constants: Base XGBoost parameters to include in every trial.

    Returns:
        Mean CV metric value (MAE or configured eval_metric) from the final boosting round.
    """
    # Dynamically build the params dict based on the HYPERPARAMETERS configuration


    params = _suggest_params(trial)

    # Run CV and pruning in a single call to XGBoost's cv function
    booster_collector = BoosterCollector()
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv=True, period=monitor_periods),
        xgb.callback.EarlyStopping(rounds=early_stopping_rounds),
        XGBoostPruningCallback(trial, observation_key=f"test-{metric}"), 
        # The observation_key in the pruning callback reads from XGBoost's internal evals_log
        # dict (keyed as "test-<metric>") - not from the final DataFrame column names ("test-<metric>-mean").
        booster_collector,
    ]

    history = xgb.cv(params={**xgboost_constants, **params},
                     dtrain=xgb.DMatrix(X, label=y, enable_categorical=True),
                     num_boost_round=boosting_rounds,
                     folds = list(cv.split(X)),
                     metrics=[metric],
                     callbacks=callbacks,
                     seed=C.SEED)

    _log_trial_cv_results(trial=trial, metric=metric, history=history)

    _save_trial_artifacts(study_name=study_name,
                          trial=trial,
                          history=history,
                          boosters=booster_collector.cvfolds)

    return history[f"test-{metric}-mean"].values[-1]


def nested_cv(
        X: pd.DataFrame, 
        y: pd.Series) -> None:
    """
    Perform nested cross-validation with Optuna hyperparameter tuning and XGBoost.
    The outer loop uses TimeSeriesCV to create train/test splits, and the inner loop performs hyperparameter tuning with another TimeSeriesCV for each trial.
    Results and artifacts are logged to disk and the Optuna database for later analysis.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).

    Returns:
        None (results are logged to disk and Optuna DB).
    """

    cv = TimeSeriesCV(n_splits=C.N_OUTER_SPLITS,
                      horizon=C.FORECAST_HORIZON,
                      train_size=C.OUTER_TRAIN_SIZE)

    for fold, (outer_train_idx, outer_test_idx) in enumerate(cv.split(X), start=1):
        study_name = f"study_fold_{fold}"
        Xt, yt = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        Xv, yv = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]
        train_dm = xgb.DMatrix(Xt, yt, enable_categorical=True)
        test_dm  = xgb.DMatrix(Xv, enable_categorical=True)
        logger.info(f"Outer fold {fold}/{C.N_OUTER_SPLITS}  (train={len(Xt)} samples, test={len(Xv)} samples)")

        logger.info(f"Running hyperparameter tuning ({C.NUM_TRIALS} trials, {C.N_INNER_SPLITS} inner folds)...")
        inner_cv = TimeSeriesCV(n_splits=C.N_INNER_SPLITS,
                                horizon=C.FORECAST_HORIZON,
                                train_size=C.INNER_TRAIN_SIZE)
        obj_fcn = lambda trial: _objective(trial, Xt, yt, inner_cv, study_name=study_name)
        study   = optuna.create_study(study_name=study_name,
                                    storage=C.STORAGE_URL,
                                    load_if_exists=True,
                                    direction="minimize",
                                    pruner=MedianPruner(n_startup_trials=C.NUM_STARTUP_TRIALS),
                                    sampler=TPESampler(seed=C.SEED))
        study.optimize(obj_fcn, n_trials=C.NUM_TRIALS, n_jobs=-1)
        _log_study(study)

        logger.info(f"Training final model on outer fold {fold} with the best hyperparameters...")
        final_booster = xgb.train(params={**C.XGB_CONSTANTS, **study.best_trial.params},
                                num_boost_round=study.best_trial.user_attrs["best_n_rounds"],
                                dtrain=train_dm)

        logger.info(f"Evaluating final model on outer fold {fold} test set...")
        preds = final_booster.predict(test_dm)
        metrics = compute_metrics(yv.values, preds)
        metrics_filename = os.path.join(C.LOG_DIR, study_name, f"test_set_metrics_fold_{fold}.csv")
        metrics.to_csv(metrics_filename)

    logger.info("Nested cross-validation complete. All results logged.")

    return 