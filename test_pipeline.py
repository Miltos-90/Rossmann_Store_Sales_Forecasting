"""Smoke-test for the Rossmann store sales forecasting pipeline.

Generates a small synthetic dataset with a strong, easy-to-fit weekly pattern,
runs the full feature-engineering → nested CV pipeline, prints test-set metrics,
and plots real vs predicted sales for the last outer fold.

Usage: conda run -n rossmann python test_pipeline.py
"""

import os
import glob
import logging
import warnings
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

from src.features import make_features, make_targets, attach_store_data
from src.preprocessing import drop_closed, drop_null_targets
from src.engine import nested_cv
from src.engine.cv import TimeSeriesCV
from src.engine.metrics import compute_metrics
from pandas.tseries.offsets import DateOffset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SEED = 42
N_STORES = 3
DAYS     = 500
FORECAST_HORIZON = 14
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR     = os.path.join(tempfile.gettempdir(), "rossmann_test")
PLOT_DIR    = os.path.join(ROOT, "artifacts", "test_outputs")

rng  = np.random.default_rng(SEED)


cv_config = {
    "n_outer_splits":   2,
    "n_inner_splits":   2,
    "forecast_horizon": FORECAST_HORIZON,
    "outer_train_size": 150,
    "inner_train_size": 80,
}

study_config = {
    "storage_url":           f"sqlite:///{os.path.join(LOG_DIR, 'test.db')}",
    "n_trials":              8,
    "n_startup_trials":      3,
    "n_jobs":                1,
    "seed":                  SEED,
    "log_dir":               LOG_DIR,
    "num_boost_rounds":      5000,
    "early_stopping_rounds": 20,
    "monitor_periods":       50,
    "xgb_constants": {
        "tree_method":    "hist",
        "device":         "cpu",
        "multi_strategy": "multi_output_tree",
        "objective":      "reg:squarederror",
        "eval_metric":    "rmse",
        "verbosity":      0,
    },
    "hyperparameters": {
        "max_depth":        ("suggest_int",   2,    6,    {"log": False}),
        "learning_rate":    ("suggest_float", 1e-2, 0.3,  {"log": True}),
        "subsample":        ("suggest_float", 0.6,  1.0,  {"log": False}),
        "colsample_bytree": ("suggest_float", 0.6,  1.0,  {"log": False}),
        "min_child_weight": ("suggest_float", 1.0,  20.0, {"log": True}),
        "reg_alpha":        ("suggest_float", 1e-4, 1.0,  {"log": True}),
        "reg_lambda":       ("suggest_float", 1e-4, 1.0,  {"log": True}),
        "gamma":            ("suggest_float", 0.0,  5.0,  {"log": False}),
    },
}


# ── 1. Synthetic data ─────────────────────────────────────────────────────────
# 3 stores × 500 calendar days.  Strong deterministic weekly + trend signal so
# the model can overfit easily and produce low errors.

dates    = pd.date_range("2013-01-01", periods=DAYS, freq="D")
rows = []
for store_id in range(1, N_STORES + 1):
    base = 300 + store_id * 150          # stores differ in overall level
    for date in dates:
        dow      = date.dayofweek + 1    # 1=Mon … 7=Sun
        is_sun   = dow == 7
        seasonal = base + 250 * np.sin(2 * np.pi * (dow - 1) / 6)   # peak Wed
        trend    = 0.2 * (date - dates[0]).days
        promo    = int(rng.random() < 0.25)
        noise    = rng.normal(0, 15)
        sales    = max(0.0, seasonal + trend + promo * 100 + noise) if not is_sun else 0.0
        rows.append({
            "Date":          date,
            "Store":         store_id,
            "DayOfWeek":     dow,
            "Open":          0 if is_sun else 1,
            "Promo":         promo,
            "StateHoliday":  "0",
            "SchoolHoliday": 0,
            "Sales":         sales,
        })

train = pd.DataFrame(rows)

# ── 2. Store metadata ─────────────────────────────────────────────────────────
stores = pd.DataFrame({
    "Store":               list(range(1, N_STORES + 1)),
    "StoreType":           ["a", "b", "c"],
    "Assortment":          ["a", "a", "b"],
    "CompetitionDistance": [500.0, 1200.0, 800.0],
    "CompetitionSinceDate": pd.to_datetime(["2012-03-01", "2011-06-01", "2013-01-01"]),
    "Promo2SinceDate":     pd.to_datetime(["2012-06-01", pd.NaT,       "2013-03-01"]),
    "PromoInterval":       ["Jan,Apr,Jul,Oct", None,                   "Mar,Jun,Sep,Dec"],
})

# ── 3. Feature engineering ────────────────────────────────────────────────────


# Reduced lag/roll/diff sets so feature engineering completes quickly
LAGS_TEST = [
    DateOffset(days=1), DateOffset(weeks=1),
    DateOffset(weeks=2), DateOffset(months=1),
]
DIFFS_TEST = [DateOffset(days=1), DateOffset(months=1)]
ROLL_WINDOWS_TEST = {
    7:  [DateOffset(days=1)],
    30: [DateOffset(weeks=1)],
}

print("Building features …")
df  = attach_store_data(train.copy(), stores.copy())
df  = make_features(df, lags=LAGS_TEST, roll_windows=ROLL_WINDOWS_TEST, diffs=DIFFS_TEST)

raw_sales = train[["Date", "Store", "Sales"]].copy()
targets   = make_targets(raw_sales, horizon=FORECAST_HORIZON)

X = df
y = targets.reindex(X.index)   # align to same (Date, Store) MultiIndex

X, y = drop_closed(X, y)
X, y = drop_null_targets(X, y)

print(f"X: {X.shape}   y: {y.shape}")
print(f"Unique dates: {X.index.get_level_values('Date').nunique()}   "
      f"Stores: {sorted(X.index.get_level_values('Store').unique().tolist())}")

# ── 4. CV / study config ──────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ── 5. Run nested CV ──────────────────────────────────────────────────────────
print("\nRunning nested CV …")
nested_cv(X, y, cv_config, study_config)

# ── 6. Print test-set metrics ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST SET METRICS PER OUTER FOLD")
print("=" * 60)
for csv_path in sorted(glob.glob(os.path.join(LOG_DIR, "study_fold_*", "test_set_metrics_*.csv"))):
    fold_name = os.path.basename(os.path.dirname(csv_path))
    m = pd.read_csv(csv_path, index_col=0)
    print(f"\n── {fold_name} ──────────────────")
    print(m.to_string())

# ── 7. Re-train last outer fold to obtain predictions for plotting ─────────────
print("\nGenerating forecast plot for last outer fold …")

outer_cv = TimeSeriesCV(
    n_splits=cv_config["n_outer_splits"],
    horizon=cv_config["forecast_horizon"],
    train_size=cv_config["outer_train_size"],
)
splits = list(outer_cv.split(X))
train_idx, test_idx = splits[-1]

Xt, yt = X.iloc[train_idx], y.iloc[train_idx]
Xv, yv = X.iloc[test_idx],  y.iloc[test_idx]

last_fold  = cv_config["n_outer_splits"]
study      = optuna.load_study(
    study_name=f"study_fold_{last_fold}",
    storage=study_config["storage_url"],
)
best_params   = study.best_trial.params
best_n_rounds = study.best_trial.user_attrs["best_n_rounds"]

final_model = xgb.train(
    params={**study_config["xgb_constants"], **best_params},
    dtrain=xgb.DMatrix(Xt, yt.values, enable_categorical=True),
    num_boost_round=best_n_rounds,
)
preds = final_model.predict(xgb.DMatrix(Xv, enable_categorical=True))
# preds: (n_test_rows, FORECAST_HORIZON)

# ── Per-step metrics ──────────────────────────────────────────────────────────
step_metrics = compute_metrics(yv.values, preds)
print("\n── Per-step metrics (last outer fold, re-trained model) ──")
print(step_metrics.to_string())

# ── 8. Per-timestep plots ─────────────────────────────────────────────────────
target_cols = list(yv.columns)   # ["lead_1_days", "lead_2_days", …]
n_steps     = len(target_cols)

print(f"\nSaving {n_steps} per-step forecast plots → {PLOT_DIR}")
for step_idx, col in enumerate(target_cols):
    step_num = step_idx + 1

    actual = (pd.Series(yv[col].values, index=Xv.index)
                .groupby(level="Date").mean())
    pred   = (pd.Series(preds[:, step_idx], index=Xv.index)
                .groupby(level="Date").mean())

    m = step_metrics.loc[step_num]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"Outer fold {last_fold}  |  t+{step_num} day forecast  "
        f"|  best_n_rounds={best_n_rounds}",
        fontsize=11,
    )

    ax.plot(actual.index, actual.values, "o-",  ms=4, lw=1.5, label="Actual")
    ax.plot(pred.index,   pred.values,   "x--", ms=4, lw=1.5, label="Predicted")

    ax.set_title(
        f"MAE={m['MAE']:.1f}   RMSE={m['RMSE']:.1f}   "
        f"MAPE={m['MAPE']:.1f}%   R²={m['R2']:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (mean across stores)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"step_{step_num:02d}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

# ── Summary plot: RMSE per step ───────────────────────────────────────────────
steps = step_metrics.index[step_metrics.index != "overall"]
rmses = step_metrics.loc[steps, "RMSE"].values

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(steps)), rmses, color="steelblue", alpha=0.8)
ax.axhline(step_metrics.loc["overall", "RMSE"], color="red", ls="--",
           label=f"Overall RMSE = {step_metrics.loc['overall', 'RMSE']:.1f}")
ax.set_title(f"RMSE by forecast step  |  outer fold {last_fold}")
ax.set_xlabel("Step (t + N days)")
ax.set_ylabel("RMSE")
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([str(s) for s in steps], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
summary_path = os.path.join(PLOT_DIR, "rmse_by_step.png")
plt.savefig(summary_path, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"Plots saved → {PLOT_DIR}")
