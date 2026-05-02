import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _metrics_1d(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute forecasting metrics for 1D arrays of true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, RMSE, MAPE, RMSPE, and R2 metrics.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    # sklearn metrics
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan")

    # percentage metrics
    nonzero = y_true != 0
    if nonzero.any():
        pct = (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
        mape  = float(np.mean(np.abs(pct)) * 100)
        rmspe = float(np.sqrt(np.mean(pct**2)) * 100)
    else:
        mape = rmspe = float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "RMSPE": rmspe, "R2": r2}

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute forecasting metrics per horizon step + overall.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    metrics = {
        step + 1: _metrics_1d(y_true[:, step], y_pred[:, step])
        for step in range(y_true.shape[1])
    }

    df = pd.DataFrame(metrics).T
    df.index.name = "step"
    df.loc["overall"] = df.mean()

    return df