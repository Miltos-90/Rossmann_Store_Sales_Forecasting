"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.tseries.offsets import DateOffset
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.features import make_features, make_targets
from cv import TimeSeriesCV
from callbacks import EarlyStoppingCV, LogEvalCallback
from metrics import metrics as metrics_fcn
from logger import logger

# Outer CV: performance estimation
N_OUTER_SPLITS  = 2
OUTER_TRAIN_SIZE = 180  # days
METRICS_FILE = os.path.join("./artifacts", "metrics.csv")
# NOTE: More constants in logger.py

# Inner CV: hyperparameter search
N_INNER_SPLITS  = 2
INNER_TRAIN_SIZE = 90   # days
NUM_TRIALS       = 2
LOG_PERIOD        = 20   # log CV metrics every LOG_PERIOD boosting rounds

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
    "early_stopping_rounds": 10,
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

def make_objective(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: TimeSeriesCV,
    xgb_constants: dict,
    log_period: int = LOG_PERIOD,
):
    """Return a hyperopt-compatible objective closed over the given training data.

    Parameters
    ----------
    X : pd.DataFrame
        Outer-fold training data the inner CV will be run on.
    y : pd.DataFrame
        Outer-fold target data the inner CV will be run on.
    cv : TimeSeriesCV
        Pre-configured inner cross-validator.
    xgb_constants : dict
        Fixed XGBoost parameters that are not part of the search space.
    log_period : int
        Log CV metrics every ``log_period`` boosting rounds.

    Returns
    -------
    Callable[[dict], dict]
        Objective function that hyperopt can minimise (loss = mean RMSPE).
    """

    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    folds = list(cv.split(X))

    def objective(params: dict) -> dict:

        xgb_params = {**xgb_constants, **params}
        num_boost_round = xgb_params.pop("num_boost_round")
        early_stopping_rounds = xgb_params.pop("early_stopping_rounds")
        eval_metric = xgb_params["eval_metric"]

        # Fresh callbacks per trial to avoid stale state across hyperopt trials.
        cv_callbacks = [
            EarlyStoppingCV(rounds=early_stopping_rounds, maximize=False, metric_name=eval_metric, data_name="test"),
            LogEvalCallback(eval_metric, period=log_period),
        ]

        cv_res = xgb.cv(params=xgb_params,
                        dtrain=dtrain,
                        num_boost_round=num_boost_round,
                        folds=folds,
                        metrics=eval_metric,
                        as_pandas=True,
                        maximize=False,
                        seed=0,
                        callbacks=cv_callbacks)

        loss = cv_res[f'test-{eval_metric}-mean'].iloc[-1]

        # If early stopping triggered, best_iteration is the optimal number of rounds; otherwise, use the max.
        params["num_boost_round"] = cv_res.attrs.get("best_iteration", len(cv_res))  
        
        return {"loss": loss, "status": STATUS_OK, "params": params}

    return objective


toy_df = build_dataset()
logger.info("Toy dataset: %d rows", len(toy_df))

lags    = [DateOffset(days=d) for d in range(1, 8)]   # lags 1–7: target[t+h] = lag_{7-h}_days
diffs   = [DateOffset(days=1), DateOffset(days=7)]
windows = {7: [DateOffset(days=1)]}
horizon = 7

X = make_features(toy_df.copy(), lags=lags, roll_windows=windows, diffs=diffs)
y = make_targets(toy_df[["Date", "Store", "Sales"]], horizon=horizon)

valid = y.notna().all(axis=1)  # only keep rows where all target columns are present
X, y = X.loc[valid], y.loc[valid]
logger.info("After feature engineering: %d samples, %d features", len(X), X.shape[1])

cv = TimeSeriesCV(n_splits=N_OUTER_SPLITS, horizon=horizon, train_size=OUTER_TRAIN_SIZE)

outer_results: list[dict] = []
for fold, (outer_train_idx, outer_test_idx) in enumerate(cv.split(X), start=1):
    Xt, yt = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
    Xv, yv = X.iloc[outer_test_idx],  y.iloc[outer_test_idx]

    logger.info("Outer fold %d/%d  (train=%d samples, test=%d samples)",
                fold, N_OUTER_SPLITS, len(Xt), len(Xv))

    # ── Inner loop: hyperopt over inner CV folds ────────────────────────
    logger.info("Constructing objective function for inner CV...")
    inner_cv = TimeSeriesCV(n_splits=N_INNER_SPLITS, horizon=horizon, train_size=INNER_TRAIN_SIZE)
    objective = make_objective(Xt, yt, inner_cv, XGB_CONSTANTS)

    logger.info("Running hyperopt (%d trials, %d inner folds)...", NUM_TRIALS, N_INNER_SPLITS)
    trials = Trials()
    fmin(fn=objective,
         space=SEARCH_SPACE,
         algo=tpe.suggest,
         max_evals=NUM_TRIALS,
         trials=trials,
         verbose=False)
    
    best_params = trials.best_trial["result"]["params"].copy()
    logger.info("Best hyperparameters found:\n%s", json.dumps(best_params, indent=2))

    logger.info("Training final model on outer fold %d with best hyperparameters...", fold)
    num_boost_round = best_params.pop("num_boost_round")
    xgb_params = {**XGB_CONSTANTS, **best_params}
    train_dm = xgb.DMatrix(Xt, yt, enable_categorical=True)
    test_dm  = xgb.DMatrix(Xv, enable_categorical=True)

    final_booster = xgb.train(params=xgb_params,
                              num_boost_round=num_boost_round,
                              dtrain=train_dm)

    preds = final_booster.predict(test_dm)
    metrics = metrics_fcn(yv.values, preds)

    outer_results.append({"fold": fold, "best_params": best_params, "metrics": metrics})
    logger.info("Outer fold %d test metrics:\n%s", fold, metrics.to_string(float_format="{:.4f}".format))

logger.info("Saving metrics")
fold_frames = [r["metrics"].assign(fold=r["fold"]) for r in outer_results]
metrics_df  = pd.concat(fold_frames).reset_index()  # reset brings "step" back as a column
metrics_df.to_csv(METRICS_FILE, index=False)