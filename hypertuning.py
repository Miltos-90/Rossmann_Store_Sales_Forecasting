"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.tseries.offsets import DateOffset
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.features import make_features, make_targets
from cv import TimeSeriesCV

LOG_FILE = "hypertuning.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Outer CV: performance estimation
N_OUTER_SPLITS  = 2
OUTER_TRAIN_SIZE = 180  # days

# Inner CV: hyperparameter search
N_INNER_SPLITS  = 2
INNER_TRAIN_SIZE = 90   # days
NUM_TRIALS       = 20

# Dataset creation parameters
N_STORES = 3
N_DAYS   = 365


# Fixed XGBoost parameters (not tuned)
XGB_CONSTANTS: dict = {
    "tree_method": "hist",
    "device": "cpu",  # cpu/cuda
    "multi_strategy": "multi_output_tree",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}

SEARCH_SPACE: dict = {
    "num_boost_round":  hp.choice("num_boost_round",   [100, 200, 300, 500]),
    "max_depth":        hp.choice("max_depth",          [3, 4, 5, 6, 8]),
    "eta":              hp.loguniform("eta",             np.log(0.01), np.log(0.3)),
    "subsample":        hp.uniform("subsample",          0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree",   0.5, 1.0),
    "min_child_weight": hp.choice("min_child_weight",    [1, 3, 5, 10, 20]),
    "alpha":            hp.loguniform("alpha",           np.log(1e-4), np.log(10.0)),
    "lambda":           hp.loguniform("lambda",          np.log(1e-4), np.log(10.0)),
}

def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    n_stores, n_days = N_STORES, N_DAYS
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")

    # Stationary weekly seasonal pattern: Sales[i] = base + amplitude * sin(2π*i/7)
    # This is identical every 7 days, so lag_7_days is a near-perfect predictor and
    # the distribution is the same in every CV fold (no extrapolation problem).
    rng_params = np.random.default_rng(42)
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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common forecasting accuracy metrics.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of any shape; they are flattened before computation.
        NaN / inf values are silently dropped.

    Returns
    -------
    dict with keys: MAE, RMSE, MAPE (%), RMSPE (%), R2
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    residuals = y_true - y_pred

    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Percentage metrics — exclude zero actuals to avoid division by zero
    nonzero = y_true != 0
    if nonzero.any():
        pct_err = residuals[nonzero] / y_true[nonzero]
        mape  = float(np.mean(np.abs(pct_err)) * 100)
        rmspe = float(np.sqrt(np.mean(pct_err ** 2)) * 100)
    else:
        mape  = float("nan")
        rmspe = float("nan")

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "RMSPE": rmspe, "R2": r2}


def compute_metrics_per_step(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Compute metrics independently for each horizon step.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples, horizon)

    Returns
    -------
    pd.DataFrame with 1-based step as index and metric names as columns.
    An ``"overall"`` row (mean across steps) is appended.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true, y_pred = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)

    step_metrics = {
        step + 1: compute_metrics(y_true[:, step], y_pred[:, step])
        for step in range(y_true.shape[1])
    }
    df = pd.DataFrame(step_metrics).T
    df.index.name = "step"
    df.loc["overall"] = df.mean()
    return df


def make_objective(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    inner_cv: TimeSeriesCV,
    xgb_constants: dict,
):
    """Return a hyperopt-compatible objective closed over the given training data.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame
        Outer-fold training data the inner CV will be run on.
    inner_cv : TimeSeriesCV
        Pre-configured inner cross-validator.
    xgb_constants : dict
        Fixed XGBoost parameters that are not part of the search space.

    Returns
    -------
    Callable[[dict], dict]
        Objective function that hyperopt can minimise (loss = mean RMSPE).
    """
    def objective(params: dict) -> dict:
        num_boost_round = params.pop("num_boost_round")
        xgb_params = {**xgb_constants, **params}

        fold_rmspe: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_in_train = X_train.iloc[inner_train_idx]
            y_in_train = y_train.iloc[inner_train_idx]
            X_in_val   = X_train.iloc[inner_val_idx]
            y_in_val   = y_train.iloc[inner_val_idx]

            dtrain = xgb.DMatrix(X_in_train, y_in_train, enable_categorical=True)
            dval   = xgb.DMatrix(X_in_val,   y_in_val,   enable_categorical=True)

            booster = xgb.train(
                params=xgb_params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                evals=[(dval, "val")],
                verbose_eval=False,
            )
            preds      = booster.predict(dval)
            per_step   = compute_metrics_per_step(y_in_val.values, preds)
            # Average RMSPE across horizon steps (equal weight per step)
            step_rmspe = float(per_step.loc[per_step.index != "overall", "RMSPE"].mean())
            fold_rmspe.append(step_rmspe)

        mean_rmspe = float(np.mean(fold_rmspe))
        return {"loss": mean_rmspe, "status": STATUS_OK,
                "params": {**params, "num_boost_round": num_boost_round}}

    return objective



toy_df = build_dataset()
log.info("Toy dataset: %d rows", len(toy_df))


lags    = [DateOffset(days=d) for d in range(1, 8)]   # lags 1–7: target[t+h] = lag_{7-h}_days
diffs   = [DateOffset(days=1), DateOffset(days=7)]
windows = {7: [DateOffset(days=1)]}
horizon = 7

X = make_features(toy_df.copy(), lags=lags, roll_windows=windows, diffs=diffs)
y = make_targets(toy_df[["Date", "Store", "Sales"]], horizon=horizon)

# Align on shared (Date, Store) index; drop only rows where targets are NaN
# (XGBoost handles NaN feature values natively)
valid = y.notna().all(axis=1)
X, y = X.loc[valid], y.loc[valid]
log.info("After feature engineering: %d samples, %d features", len(X), X.shape[1])

cv = TimeSeriesCV(n_splits=N_OUTER_SPLITS, horizon=horizon, train_size=OUTER_TRAIN_SIZE)
outer_results: list[dict] = []

for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(cv.split(X), start=1):
    X_outer_train, y_outer_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
    X_outer_test,  y_outer_test  = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]

    log.info("")
    log.info("=" * 60)
    log.info("Outer fold %d/%d  (train=%d samples, test=%d samples)",
             outer_fold, N_OUTER_SPLITS, len(X_outer_train), len(X_outer_test))

    # ── Inner loop: hyperopt over inner CV folds ────────────────────────
    inner_cv = TimeSeriesCV(n_splits=N_INNER_SPLITS, horizon=horizon, train_size=INNER_TRAIN_SIZE)
    objective = make_objective(X_outer_train, y_outer_train, inner_cv, XGB_CONSTANTS)

    log.info("  Running hyperopt (%d trials, %d inner folds)...", NUM_TRIALS, N_INNER_SPLITS)
    trials = Trials()
    fmin(
        fn=objective,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=NUM_TRIALS,
        trials=trials,
        verbose=False,
    )
    best_params = trials.best_trial["result"]["params"].copy()
    num_boost_round = best_params.pop("num_boost_round")

    log.info("  Best hyperparameters:")
    for k, v in best_params.items():
        log.info("    %s: %s", k, f"{v:.6g}" if isinstance(v, float) else v)

    # ── Retrain on full outer training fold with best params ────────────
    final_params = {**XGB_CONSTANTS, **best_params}
    train_dm = xgb.DMatrix(X_outer_train, y_outer_train, enable_categorical=True)
    test_dm  = xgb.DMatrix(X_outer_test, enable_categorical=True)

    final_booster = xgb.train(
        params=final_params,
        num_boost_round=num_boost_round,
        dtrain=train_dm,
    )

    preds    = final_booster.predict(test_dm)
    per_step = compute_metrics_per_step(y_outer_test.values, preds)
    outer_results.append({"outer_fold": outer_fold,
                          "best_params": best_params,
                           "per_step": per_step})

    log.info("  Outer fold %d test metrics:\n%s", outer_fold,
             per_step.to_string(float_format="{:.4f}".format))

log.info("")
log.info("=" * 60)

log.info("Saving metrics to hypertuning_metrics.csv")
fold_frames = [r["per_step"].assign(outer_fold=r["outer_fold"]) for r in outer_results]
metrics_df  = pd.concat(fold_frames).reset_index()  # reset brings "step" back as a column

