"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import json
import logging
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import optuna_integration

from pandas.tseries.offsets import DateOffset
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study

import src.constants as C

from src.features import make_features, make_targets
from src.cv import TimeSeriesCV
from src.metrics import metrics as metrics_fcn
from src.callbacks import BoosterCollector

os.makedirs(C.LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(C.LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,  # ensure no duplicate handlers if this cell is re-run in a notebookcls
)


logger = logging.getLogger(__name__)


from sklearn.model_selection import BaseCrossValidator
from optuna.trial import Trial

def _save_trial_artifacts(
    study_name: str,
    trial: Trial,
    history: pd.DataFrame,
    boosters: list[xgb.Booster] | None,
) -> None:
    """Persist CV history and fold boosters to disk, keyed by study/trial."""
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



def log_study(study: Study) -> None:
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

    Args:
        trial: Optuna trial.
        metric: Metric name (e.g., "mae").
        history: DataFrame containing CV results for each boosting round.
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

def _suggest_params(trial: Trial) -> dict:
    params = {}
    for name, (method, low, high, kwargs) in C.HYPERPARAMETERS.items():
        suggest_fn = getattr(trial, method)
        params[name] = suggest_fn(name, low, high, **kwargs)
    return params


def objective(
    trial: Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    study_name: str = "",
) -> float:
    """Optuna objective for Stage 1: broad search over the full hyperparameter space.

    Suggests values from wide ranges for all XGBoost hyperparameters and
    delegates evaluation (including per-fold pruning) to :func:`_evaluate`.

    Args:
        trial: Active Optuna trial.
        X: Feature matrix.
        y: Target vector.
        cv: Cross-validation splitter.
        base_model: Base sklearn Pipeline to clone for each fold.

    Returns:
        Mean CV MAE for the suggested hyperparameter configuration.
    """
    # Dynamically build the params dict based on the HYPERPARAMETERS configuration


    params = _suggest_params(trial)

    # Run CV and pruning in a single call to XGBoost's cv function
    metric = C.XGB_CONSTANTS["eval_metric"]
    booster_collector = BoosterCollector()
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv=True, period=100),
        xgb.callback.EarlyStopping(rounds=C.EARLY_STOPPING_ROUNDS),
        optuna_integration.xgboost.XGBoostPruningCallback(trial, observation_key=f"test-{metric}"),  # The observation_key in the pruning callback reads from XGBoost's internal evals_log dict (keyed as "test-<metric>"), not from the final DataFrame column names ("test-<metric>-mean").
        booster_collector,
    ]

    history = xgb.cv(params={**C.XGB_CONSTANTS, **params},
                     dtrain=xgb.DMatrix(X, label=y, enable_categorical=True),
                     num_boost_round=C.NUM_BOOST_ROUNDS,
                     folds = list(cv.split(X)),
                     metrics=[metric],
                     callbacks=callbacks,
                     seed=C.SEED)

    _log_trial_cv_results(trial=trial, metric=metric, history=history)
    _save_trial_artifacts(
        study_name=study_name,
        trial=trial,
        history=history,
        boosters=booster_collector.cvfolds,
    )

    return history[f"test-{metric}-mean"].values[-1]


##############################################
def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    n_stores, n_days = 3, 365*3
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")

    # Stationary weekly seasonal pattern: Sales[i] = base + amplitude * sin(2π*i/7)
    # This is identical every 7 days, so lag_7_days is a near-perfect predictor and
    # the distribution is the same in every CV fold (no extrapolation problem).
    rng_params      = np.random.default_rng(42)
    store_base      = {s: rng_params.uniform(3000.0, 10000.0) for s in range(1, n_stores + 1)}
    store_amplitude = {s: rng_params.uniform(200.0,   2000.0) for s in range(1, n_stores + 1)}

    toy_rows = []
    for s in range(1, n_stores + 1):
        for i, d in enumerate(dates):
            toy_rows.append({
                "Date": d, "Store": s,
                "DayOfWeek": d.dayofweek + 1,
                "Open": 1,
                "Promo": rng.integers(0, 2),
                "Promo2": rng.integers(0, 2),
                "Promo2SinceDate": pd.NaT,
                "StateHoliday": "0",
                "SchoolHoliday": int(rng.integers(0, 2)),
                "StoreType": rng.choice(["a", "b", "c", "d"]),
                "Assortment": rng.choice(["a", "b", "c"]),
                "CompetitionDistance": rng.uniform(100, 5000),
                "CompetitionSinceDate": pd.Timestamp("2010-01-01"),
            })

    toy_df = pd.DataFrame(toy_rows)
    toy_df["Sales"] = toy_df.apply(
        lambda r: store_base[r["Store"]] + store_amplitude[r["Store"]] *
                  np.sin(2 * np.pi * int(np.where(dates == r["Date"])[0][0]) / 7),
        axis=1,
    )

    return toy_df

toy_df = build_dataset()
logger.info("Toy dataset: %d rows", len(toy_df))
##############################################


X = make_features(toy_df.copy(), lags=C.LAGS, roll_windows=C.ROLL_WINDOWS, diffs=C.DIFFS)
y = make_targets(toy_df[["Date", "Store", "Sales"]], horizon=C.FORECAST_HORIZON)
valid = y.notna().all(axis=1)  # only keep rows where all target columns are present
X, y = X.loc[valid], y.loc[valid]
logger.info(f"Feature engineering: {len(X)} samples, {X.shape[1]} features")

results: list[dict] = []
cv = TimeSeriesCV(n_splits=C.N_OUTER_SPLITS,
                  horizon=C.FORECAST_HORIZON,
                  train_size=C.OUTER_TRAIN_SIZE)

for fold, (outer_train_idx, outer_test_idx) in enumerate(cv.split(X), start=1):
    Xt, yt = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
    Xv, yv = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]
    train_dm = xgb.DMatrix(Xt, yt, enable_categorical=True)
    test_dm  = xgb.DMatrix(Xv, enable_categorical=True)

    logger.info(f"Outer fold {fold}/{C.N_OUTER_SPLITS}  (train={len(Xt)} samples, test={len(Xv)} samples)")

    inner_cv = TimeSeriesCV(n_splits=C.N_INNER_SPLITS,
                            horizon=C.FORECAST_HORIZON,
                            train_size=C.INNER_TRAIN_SIZE)
    
    study_name = f"study_fold_{fold}"
    obj_fcn = lambda trial: objective(trial, Xt, yt, inner_cv, study_name=study_name)

    logger.info(f"Running hyperparameter tuning ({C.NUM_TRIALS} trials, {C.N_INNER_SPLITS} inner folds)...")
    pruner  = MedianPruner(n_startup_trials=0, n_warmup_steps=0, interval_steps=1)  # no startup/warmup for Stage 2
    sampler = TPESampler(seed=C.SEED)
    study   = optuna.create_study(study_name=study_name,
                                  storage=C.STORAGE_URL,
                                  load_if_exists=True,
                                  direction="minimize",
                                  pruner=pruner,
                                  sampler=sampler)
    study.optimize(obj_fcn, n_trials=C.NUM_TRIALS, n_jobs=-1)
    log_study(study)

    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_n_rounds = best_trial.user_attrs["best_n_rounds"]
    logger.info("Best hyperparameters:\n%s", json.dumps(best_params, indent=2))
    logger.info(f"Training final model on outer fold {fold} with the best hyperparameters...")
    xgb_params = C.XGB_CONSTANTS
    xgb_params.update(best_params)
    final_booster = xgb.train(params=xgb_params,
                              num_boost_round=best_n_rounds,
                              dtrain=train_dm)

    logger.info(f"Evaluating final model on outer fold {fold} test set...")
    preds = final_booster.predict(test_dm)
    metrics = metrics_fcn(yv.values, preds)
    metrics_filename = os.path.join(C.LOG_DIR, study_name, f"test_set_metrics_fold_{fold}.csv")
    metrics.to_csv(metrics_filename)

