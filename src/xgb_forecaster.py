"""XGBForecaster: XGBoost-based multi-step time-series forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Optional


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "device": "cuda",           # GPU acceleration; pass {"device": "cpu"} to override
    # multi_strategy="multi_output_tree" (vector leaf) is not yet supported on GPU;
    # omitting it uses the default "one_output_per_tree" which works on both CPU and GPU.
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 30,
    "verbosity": 0,
}


class XGBForecaster:
    """XGBoost-based multi-step time-series forecaster.

    Parameters
    ----------
    horizon : int
        Number of future periods to forecast simultaneously (width of y).
    params : dict, optional
        XGBoost hyper-parameters that override the defaults.
        Pass ``{"device": "cpu"}`` when no GPU is available.
    """

    def __init__(self, horizon: int, params: Optional[dict] = None) -> None:
        self.horizon = horizon
        self.params: dict = {**_DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame,
        eval_set: Optional[list[tuple]] = None,
    ) -> xgb.XGBRegressor:
        """Fit an XGBoost multi-output regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples, horizon) or (n_samples,)
        eval_set : list of (X_val, y_val), optional
            When provided, early stopping is applied against the first entry.

        Returns
        -------
        xgb.XGBRegressor
            The fitted model (also stored as ``self.model``).
        """
        params = dict(self.params)
        if eval_set is None:
            # early_stopping_rounds requires an eval_set; remove it when absent
            params.pop("early_stopping_rounds", None)

        self.model = xgb.XGBRegressor(**params)

        fit_kwargs: dict = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        self.model.fit(X, y, **fit_kwargs)
        return self.model

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, horizon) or (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)
