"""Smoke-test for src/engine.py using small dummy data."""

import numpy as np
from src.engine import compute_metrics, cross_validate
from src.xgb_forecaster import XGBForecaster

rng = np.random.default_rng(42)

N_SAMPLES  = 5000   # number of time steps
N_FEATURES = 3    # number of input features
HORIZON    = 7     # forecast 7 steps ahead

# Smooth random-walk features so XGBoost can learn something meaningful
X_dummy = np.cumsum(rng.standard_normal((N_SAMPLES, N_FEATURES)), axis=0)

# Targets: linear combination of features + tiny noise (multi-output)
W = rng.standard_normal((N_FEATURES, HORIZON))
y_dummy = X_dummy @ W + rng.standard_normal((N_SAMPLES, HORIZON)) * 0.1

# ---------------------------------------------------------------------------
# 1. Basic fit / predict on a held-out split
# ---------------------------------------------------------------------------
forecaster = XGBForecaster(
    horizon=HORIZON,
    params={"device": "cuda", "n_estimators": 500},  # set device="cuda" for GPU
)

train_size = int(0.8 * N_SAMPLES)
model = forecaster.fit(X_dummy[:train_size], y_dummy[:train_size])
preds = forecaster.predict(X_dummy[train_size:])

print("Hold-out metrics:")
hold_out_metrics = compute_metrics(y_dummy[train_size:], preds)
for k, v in hold_out_metrics.items():
    print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# 2. Time-series cross-validation
# ---------------------------------------------------------------------------
print("\nTime-series cross-validation (3 folds):")
cv_forecaster = XGBForecaster(
    horizon=HORIZON,
    params={"device": "cpu", "n_estimators": 100},
)
cv_results = cross_validate(cv_forecaster, X_dummy, y_dummy, n_splits=3)

print("\nCV mean metrics:")
for k, v in cv_results["mean"].items():
    print(f"  {k}: {v:.4f} +/- {cv_results['std'][k]:.4f}")
