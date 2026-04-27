"""Training engine: metrics and time-series cross-validation."""

from __future__ import annotations

import numpy as np

from sklearn.model_selection import TimeSeriesSplit

from .xgb_forecaster import XGBForecaster
from .utils import make_dmatrix


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


def cross_validate(
    forecaster: XGBForecaster,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    batch_size: int | None = None,
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
    X : np.ndarray
        Feature matrix. Row order must match the temporal order of the time series.
    y : np.ndarray
        Labels. Row order must match X.
    n_splits : int
        Number of CV folds.
    batch_size : int, optional
        Passed to ``make_dmatrix``; when set, training folds use a
        QuantileDMatrix rather than a plain DMatrix.

    Returns
    -------
    dict
        ``fold_metrics`` — list of per-fold metric dicts.
        ``mean``         — metric means across folds.
        ``std``          — metric standard deviations across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(X))), start=1):
        train_dm = make_dmatrix(X[train_idx], y[train_idx], batch_size)
        val_dm   = make_dmatrix(X[val_idx])

        # Fit without early stopping — val fold must stay unseen during training
        forecaster.fit(train_dm)
        preds = forecaster.predict(val_dm)

        metrics = compute_metrics(y[val_idx], preds)
        fold_metrics.append(metrics)
        print(f"  Fold {fold}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    keys = list(fold_metrics[0].keys())
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std  = {k: float(np.std( [m[k] for m in fold_metrics])) for k in keys}

    return {"fold_metrics": fold_metrics, "mean": mean, "std": std}
