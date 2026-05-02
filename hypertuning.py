"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.tseries.offsets import DateOffset
from hyperopt import fmin, tpe, Trials

import src.constants as C
import src.logger

from src.objective import make_objective
from src.features import make_features, make_targets
from src.cv import TimeSeriesCV
from src.metrics import metrics as metrics_fcn


logger = logging.getLogger(__name__)

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
    objective = make_objective(Xt, yt, inner_cv, C.XGB_CONSTANTS)

    logger.info(f"Running hyperopt ({C.NUM_TRIALS} trials, {C.N_INNER_SPLITS} inner folds)...")
    trials = Trials()
    fmin(fn=objective,
         space=C.SEARCH_SPACE,
         algo=tpe.suggest,
         max_evals=C.NUM_TRIALS,
         trials=trials,
         verbose=False)
    
    best_params = trials.best_trial["result"]["params"].copy()
    logger.info("Best hyperparameters found:\n%s", json.dumps(best_params, indent=2))

    logger.info(f"Training final model on outer fold {fold} with best hyperparameters...")
    xgb_params = {**C.XGB_CONSTANTS, **best_params}
    num_boost_round = xgb_params.pop("num_boost_round")
    final_booster = xgb.train(params=xgb_params,
                              num_boost_round=num_boost_round,
                              dtrain=train_dm)

    logger.info(f"Evaluating final model on outer fold {fold} test set...")
    preds = final_booster.predict(test_dm)
    metrics = metrics_fcn(yv.values, preds)
    r = {"fold": fold, "best_params": best_params, "metrics": metrics}
    results.append(r)
    logger.info(f"Outer fold {fold} test metrics:\n{metrics.to_string(float_format='{:.4f}'.format)}")

logger.info("Saving metrics")
fold_frames = [r["metrics"].assign(fold=r["fold"]) for r in results]
metrics_df  = pd.concat(fold_frames).reset_index()  # reset brings "step" back as a column
metrics_df.to_csv(C.METRICS_FILE, index=False)