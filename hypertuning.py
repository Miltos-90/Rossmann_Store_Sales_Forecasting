"""Nested time-series cross-validation with hyperopt for XGBForecaster."""

import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from typing import Callable

from src.xgb_forecaster import XGBForecaster
from src.engine import compute_metrics, cross_validate

import numpy as np
import xgboost as xgb
from src.engine import compute_metrics, cross_validate
from src.xgb_forecaster import XGBForecaster
from src.utils import make_dmatrix


def build_dataset(seed, n_samples, n_features, horizon) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Smooth random-walk features so XGBoost can learn something meaningful
    X_dummy = np.cumsum(rng.standard_normal((n_samples, n_features)), axis=0)

    # Targets: linear combination of features + tiny noise (multi-output)
    W = rng.standard_normal((n_features, horizon))
    y_dummy = X_dummy @ W + rng.standard_normal((n_samples, horizon)) * 0.1

    return X_dummy, y_dummy

SEARCH_SPACE: dict = {
    "n_estimators":    hp.choice("n_estimators",    [100, 200, 300, 500]),
    "max_depth":       hp.choice("max_depth",        [3, 4, 5, 6, 8]),
    "learning_rate":   hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample":       hp.uniform("subsample",        0.6, 1.0),
    "colsample_bytree":hp.uniform("colsample_bytree", 0.5, 1.0),
    "min_child_weight":hp.choice("min_child_weight",  [1, 3, 5, 10, 20]),
    "reg_alpha":       hp.loguniform("reg_alpha",      np.log(1e-4), np.log(10.0)),
    "reg_lambda":      hp.loguniform("reg_lambda",     np.log(1e-4), np.log(10.0)),
}

# Default XGBoost parameters that are not being tuned
XGB_CONSTANTS: dict = {
    "early_stopping_rounds": 30,
    "tree_method": "hist",
    "device": "cuda", # "cpu"
    # multi_strategy="multi_output_tree" (vector leaf) is not yet supported on GPU;
    # omitting it uses the default "one_output_per_tree" which works on both CPU and GPU.
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}


SEED = 42
N_SAMPLES  = 5000   # number of time steps
N_FEATURES = 3      # number of input features
HORIZON    = 7      # forecast 7 steps ahead
NUM_OUTER_SPLITS = 3
NUM_INNER_SPLITS = 3
NUM_TRIALS = 2
BATCH_SIZE = 1024


def _make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_inner_splits: int,
    batch_size: int | None,
    constant_params: dict,
) -> Callable:
    """Return a hyperopt objective function that runs inner-fold CV."""

    def objective(trial_hyperparams: dict) -> dict:

        # Combine trial hyperparameters with constants to form the full XGBoost params dict
        xgb_params = {**constant_params, **trial_hyperparams}
        forecaster = XGBForecaster(params=xgb_params)
        cv_result = cross_validate(forecaster, X_train, y_train,
                                   n_splits=n_inner_splits, batch_size=batch_size)
        mean_mae = cv_result["mean"]["MAE"]

        # Return params in the "result" field of the best trial so they can be retrieved after fmin() 
        # finishes directly.
        return {"loss": mean_mae, "status": STATUS_OK, "params": xgb_params}

    return objective


X_dummy, y_dummy = build_dataset(SEED, N_SAMPLES, N_FEATURES, HORIZON)
outer_tscv = TimeSeriesSplit(n_splits=NUM_OUTER_SPLITS)
results: list[dict] = []

for fold, (train_idx, test_idx) in enumerate(outer_tscv.split(np.arange(len(X_dummy))), start=1):
    print(f"\n{'='*60}")
    print(f"Outer fold {fold}/{NUM_OUTER_SPLITS}  "
        f"(train={len(train_idx)}, test={len(test_idx)})")
    print(f"{'='*60}")

    train_dm = make_dmatrix(X_dummy[train_idx], y_dummy[train_idx], BATCH_SIZE)
    test_dm  = make_dmatrix(X_dummy[test_idx])

    # ── Inner loop: hyperopt ────────────────────────────────────────────────
    print(f"  Running hyperopt ({NUM_TRIALS} trials, {NUM_INNER_SPLITS} inner folds)...")
    objective = _make_objective(X_dummy[train_idx], y_dummy[train_idx], NUM_INNER_SPLITS, BATCH_SIZE, XGB_CONSTANTS)
    trials = Trials()
    fmin(
        fn=objective,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=NUM_TRIALS,
        trials=trials,
        verbose=False,
    )
    best_params = trials.best_trial["result"]["params"]

    print(f"  Best hyperparameters:")
    for k, v in best_params.items():
        if k not in ("device", "verbosity"):
            print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")

    # ── Retrain on full outer training fold ─────────────────────────
    print(f"  Retraining on full outer training fold...")
    final_forecaster = XGBForecaster(params=best_params)
    final_forecaster.fit(train_dm)

    # ── Evaluate on outer test fold ─────────────────────────────────────────────
    preds  = final_forecaster.predict(test_dm)
    y_test = y_dummy[test_idx]
    metrics = compute_metrics(y_test, preds)

    print(f"  Outer fold {fold} test metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    results.append({
        "outer_fold":  fold,
        "best_params": best_params,
        "metrics":     metrics,
    })

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Nested CV summary (mean across outer folds):")
metric_keys = list(results[0]["metrics"].keys())
for k in metric_keys:
    vals = [r["metrics"][k] for r in results]
    print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")