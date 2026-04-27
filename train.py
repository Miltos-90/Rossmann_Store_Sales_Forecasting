"""Self-contained train/test script: GPU training with categorical data, batched DMatrix."""

import json
import math

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED           = 42
N_SAMPLES      = 2000
N_FEATURES     = 3
N_CAT_FEATURES = 3   # store_type (4 levels), assortment (3 levels), day_of_week (7 levels)
HORIZON        = 7
TRAIN_FRAC     = 0.8
BATCH_SIZE     = 128

PARAMS = {
    "n_estimators":          300,
    "early_stopping_rounds": 30,
    "max_depth":             5,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "min_child_weight":      5,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "tree_method":           "hist",
    "device":                "cuda",
    "objective":             "reg:squarederror",
    "eval_metric":           "rmse",
    "verbosity":             0,
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(
    seed, n_samples, n_features, n_cat_features, horizon
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    X_cont = np.cumsum(rng.standard_normal((n_samples, n_features)), axis=0)
    W = rng.standard_normal((n_features, horizon))
    y_arr = X_cont @ W + rng.standard_normal((n_samples, horizon)) * 0.1

    cont_cols = [f"num_{i}" for i in range(n_features)]
    cat_names  = ["store_type", "assortment", "day_of_week"]
    cat_levels = (4, 3, 7)

    X = pd.DataFrame(X_cont.astype(float), columns=cont_cols)
    for name, lvl in zip(cat_names, cat_levels):
        X[name] = pd.Categorical(rng.integers(0, lvl, size=n_samples))

    y = pd.DataFrame(y_arr.astype(float), columns=[f"target_{h}" for h in range(horizon)])

    feature_types = ["q"] * n_features + ["c"] * n_cat_features
    return X, y, feature_types


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def make_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int
):
    """Yield (X_batch, y_batch) slices of ``batch_size`` rows."""
    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], y[start:end]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: XGBRegressor, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return RMSE and RMSPE for the given split."""
    dm     = xgb.DMatrix(X, enable_categorical=True) 
    y_pred = np.asarray(model.get_booster().predict(dm), dtype=float)
    y_true    = np.asarray(y, dtype=float)
    residuals = (y_true - y_pred).ravel()
    y_flat    = y_true.ravel()

    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    nonzero = y_flat != 0
    pct_err = residuals[nonzero] / y_flat[nonzero]
    rmspe   = float(np.sqrt(np.mean(pct_err ** 2)) * 100)

    return {"RMSE": rmse, "RMSPE (%)": rmspe}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- GPU verification ---
    build = xgb.build_info()
    cuda_available = build.get("USE_CUDA", False)
    print(f"XGBoost build info: USE_CUDA={cuda_available}")
    if not cuda_available:
        print("WARNING: XGBoost was not built with CUDA — training will run on CPU.")

    print("Building dataset …")
    X, y, feature_types = build_dataset(
        SEED, N_SAMPLES, N_FEATURES, N_CAT_FEATURES, HORIZON
    )

    print(f"Dataset shapes: X={X.shape}, y={y.shape}")
    print(X.dtypes)

    split   = int(len(X) * TRAIN_FRAC)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Horizon: {HORIZON}")

    print(f"\nTraining (batch_size={BATCH_SIZE}, sklearn API) …")
    params = PARAMS.copy()
    n_estimators          = int(params.pop("n_estimators"))
    early_stopping_rounds = int(params.pop("early_stopping_rounds"))
    n_batches       = math.ceil(len(X_train) / BATCH_SIZE)
    trees_per_batch = max(1, n_estimators // n_batches)

    booster = None  # Initial booster for warm start; will be updated after each batch
    iterator = make_batches(X_train, y_train, BATCH_SIZE)
    for i, (X_batch, y_batch) in enumerate(iterator, start=1):

        # Initialize model 
        is_last = i == n_batches
        model = XGBRegressor(n_estimators=trees_per_batch,
                             feature_types=feature_types,
                             enable_categorical=True,
                             early_stopping_rounds=early_stopping_rounds if is_last else None, #  only apply early stopping on the last batch
                             **params)

        # Fit on current batch, using previous booster for warm start if available
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False, "xgb_model": booster}
        model.fit(X_batch, y_batch, **fit_kwargs)

        # Keep booster for next batch
        booster = model.get_booster()

        # Log batch progress
        cfg     = json.loads(booster.save_config())
        device  = cfg.get("learner", {}).get("generic_param", {}).get("device", "unknown")
        print(f"  Batch {i:2d}/{n_batches:3d} - X shape: {X_batch.shape} - y shape: {y_batch.shape} - cumulative trees: {booster.num_boosted_rounds()} - device: {device}")

    # Final evaluation
    train_metrics = evaluate(model, X_train, y_train)
    val_metrics   = evaluate(model, X_val,   y_val)

    print("\nTrain:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
    print("Val  :", {k: f"{v:.4f}" for k, v in val_metrics.items()})
