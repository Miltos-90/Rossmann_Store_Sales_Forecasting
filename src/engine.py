"""Training engine: metrics and time-series cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from .xgb_forecaster import XGBForecaster


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    pct_err = residuals[nonzero] / y_true[nonzero]
    mape  = float(np.mean(np.abs(pct_err)) * 100)
    rmspe = float(np.sqrt(np.mean(pct_err ** 2)) * 100)

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "RMSPE": rmspe, "R2": r2}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    forecaster: XGBForecaster,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """Time-series cross-validation using ``sklearn.model_selection.TimeSeriesSplit``.

    Each fold trains on all historical data up to a cutoff and evaluates on
    the immediately following block, preserving temporal order. Early stopping
    is disabled during CV to avoid the validation fold influencing training.

    Parameters
    ----------
    forecaster : XGBForecaster
        Forecaster instance whose ``fit`` and ``predict`` methods are called
        on each fold.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples, horizon) or (n_samples,)
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict
        ``fold_metrics`` — list of per-fold metric dicts.
        ``mean``         — metric means across folds.
        ``std``          — metric standard deviations across folds.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr), start=1):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # Fit without early stopping — val fold must stay unseen during training
        forecaster.fit(X_train, y_train)
        preds = forecaster.predict(X_val)

        metrics = compute_metrics(y_val, preds)
        fold_metrics.append(metrics)
        print(f"  Fold {fold}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    keys = list(fold_metrics[0].keys())
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std  = {k: float(np.std( [m[k] for m in fold_metrics])) for k in keys}

    return {"fold_metrics": fold_metrics, "mean": mean, "std": std}
