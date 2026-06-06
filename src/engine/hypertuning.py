"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import logging
import os
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from typing import Any
from sklearn.model_selection import BaseCrossValidator
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration.xgboost import XGBoostPruningCallback

from .callbacks import BoosterCollector
from .cv import TimeSeriesCV
from .metrics import metrics_2d
from . import utils


logger = logging.getLogger(__name__)

def _suggest_params(trial: Trial, hyperparameters: dict) -> dict[str, Any]:
    """Suggest hyperparameters for a trial using Optuna's sampling methods.

    Dynamically builds a hyperparameter dictionary based on the provided
    hyperparameters configuration, using the appropriate Optuna suggest method
    for each parameter.

    Args:
        trial: Active Optuna trial.
        hyperparameters: Search space definition mapping parameter names to
            (method, low, high, kwargs) tuples.

    Returns:
        Dictionary mapping hyperparameter names to suggested values.
    """
    params = {}
    for name, (method, low, high, kwargs) in hyperparameters.items():
        suggest_fn = getattr(trial, method)
        params[name] = suggest_fn(name, low, high, **kwargs)
    return params


def _objective(
    trial: Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    study_name: str,
    study_config: dict,
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
        study_config: Optuna study settings (see nested_cv for key descriptions).

    Returns:
        Mean CV metric value (MAE or configured eval_metric) from the final boosting round.
    """
    metric = study_config["xgb_constants"]["eval_metric"]
    params = _suggest_params(trial, study_config["hyperparameters"])

    # Run CV and pruning in a single call to XGBoost's cv function
    booster_collector = BoosterCollector()
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv=True, period=study_config["monitor_periods"]),
        xgb.callback.EarlyStopping(rounds=study_config["early_stopping_rounds"]),
        XGBoostPruningCallback(trial, observation_key=f"test-{metric}"),
        # The observation_key in the pruning callback reads from XGBoost's internal evals_log
        # dict (keyed as "test-<metric>") - not from the final DataFrame column names ("test-<metric>-mean").
        booster_collector,
    ]

    all_params = {**study_config["xgb_constants"], **params}
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    num_boost_round = study_config["num_boost_rounds"]
    folds = list(cv.split(X))

    history = xgb.cv(params=all_params,
                     dtrain=dtrain,
                     num_boost_round=num_boost_round,
                     folds=folds,
                     metrics=[metric],
                     callbacks=callbacks,
                     seed=study_config["seed"])

    utils.log_trial_cv_results(trial=trial, metric=metric, history=history)

    utils.save_trial_artifacts(study_name=study_name,
                               trial=trial,
                               history=history,
                               boosters=booster_collector.cvfolds,
                               log_dir=study_config["log_dir"])

    return history[f"test-{metric}-mean"].min()


def nested_cv(
        X: pd.DataFrame,
        y: pd.DataFrame,
        cv_config: dict,
        study_config: dict,
) -> None:
    """
    Perform nested cross-validation with Optuna hyperparameter tuning and XGBoost.
    The outer loop uses TimeSeriesCV to create train/test splits, and the inner loop performs
    hyperparameter tuning with another TimeSeriesCV for each trial.
    Results and artifacts are logged to disk and the Optuna database for later analysis.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        cv_config: Cross-validation settings with keys:
            - n_outer_splits: Number of outer CV folds.
            - n_inner_splits: Number of inner CV folds for hyperparameter tuning.
            - forecast_horizon: Number of days to forecast.
            - outer_train_size: Training window size (days) for outer CV.
            - inner_train_size: Training window size (days) for inner CV.
        study_config: Optuna study settings with keys:
            - storage_url: Optuna storage URL.
            - n_trials: Number of Optuna trials per outer fold.
            - n_startup_trials: Trials before pruning is enabled.
            - n_jobs: Parallel jobs for Optuna.
            - seed: Random seed for reproducibility.
            - xgb_constants: Base XGBoost parameters for all trials and final training.

    Returns:
        None (results are logged to disk and Optuna DB).
    """

    y_log = y.apply(np.log1p)

    cv = TimeSeriesCV(n_splits=cv_config["n_outer_splits"],
                      horizon=cv_config["forecast_horizon"],
                      train_size=cv_config["outer_train_size"])

    for fold, (outer_train_idx, outer_test_idx) in enumerate(cv.split(X), start=1):
        logger.info(f"Outer fold {fold}/{cv_config['n_outer_splits']} starting...")

        # Make fold-specific artifact paths for boosters and metrics
        logger.info(f"Setting up artifact paths for fold {fold}...")
        study_name = f"study_fold_{fold}"
        artifact_dir = os.path.join(study_config["log_dir"], study_name)
        booster_filename = os.path.join(artifact_dir, f"booster_fold_{fold}.model")
        metrics_filename = os.path.join(artifact_dir, f"test_set_metrics_fold_{fold}.csv")
        predictions_filename = os.path.join(artifact_dir, f"predicted_sales_fold_{fold}.csv")
        actual_filename = os.path.join(artifact_dir, f"actual_sales_fold_{fold}.csv")
        os.makedirs(artifact_dir, exist_ok=True)

        # Prepare fold-specific datamatrices
        Xt, ylog_t = X.iloc[outer_train_idx], y_log.iloc[outer_train_idx]
        Xv, ylog_v = X.iloc[outer_test_idx],  y_log.iloc[outer_test_idx]
        train_dm = xgb.DMatrix(Xt, ylog_t, enable_categorical=True)
        test_dm  = xgb.DMatrix(Xv, enable_categorical=True)
        logger.info(f"Outer fold {fold} | Train samples: {len(Xt)}, Test samples: {len(Xv)}")

        # Run Optuna hyperparameter tuning with nested CV in the inner loop
        logger.info(f"Running hyperparameter tuning ({study_config['n_trials']} trials," 
                    f" {cv_config['n_inner_splits']} inner folds)...")
        inner_cv = TimeSeriesCV(n_splits=cv_config["n_inner_splits"],
                                horizon=cv_config["forecast_horizon"],
                                train_size=cv_config["inner_train_size"])
        pruner = MedianPruner(n_startup_trials=study_config["n_startup_trials"])
        sampler = TPESampler(seed=study_config["seed"])
        obj_fcn = lambda trial: _objective(
            trial, Xt, ylog_t, inner_cv,
            study_name=study_name,
            study_config=study_config,
        )
        study = optuna.create_study(study_name=study_name,
                                    storage=study_config["storage_url"],
                                    load_if_exists=True,
                                    direction="minimize",
                                    pruner=pruner,
                                    sampler=sampler)
        study.optimize(obj_fcn, n_trials=study_config["n_trials"], n_jobs=study_config["n_jobs"])
        utils.log_study(study)

        # Train final model on the entire outer fold training set with the best hyperparameters
        logger.info(f"Training final model on outer fold {fold} with the best hyperparameters...")
        params ={**study_config["xgb_constants"], **study.best_trial.params}
        num_boost_round = study.best_trial.user_attrs["best_n_rounds"]
        booster = xgb.train(params=params, num_boost_round=num_boost_round, dtrain=train_dm)

        # Evaluate final model on the outer fold test set and log metrics
        logger.info(f"Evaluating final model on outer fold {fold} test set...")
        preds = booster.predict(test_dm)
        is_closed = Xv['Open'] == 0
        preds = set_zero_sales_on_closed(preds, is_closed)
        preds = pd.DataFrame(data=np.expm1(preds),  # Invert log1p transformation
                             index=Xv.index,
                             columns=yv.columns)
        yv = ylog_v.apply(np.expm1)  # Invert log1p transformation for actuals

        metrics = metrics_2d(yv, preds)
        metrics.to_csv(metrics_filename)
        booster.save_model(booster_filename)
        preds.to_csv(predictions_filename)
        yv.to_csv(actual_filename)
        utils.plot_results(yv, preds)
        plt.savefig(os.path.join(artifact_dir, "predictions.png"))

    logger.info("Nested cross-validation complete. All results logged.")

    return 