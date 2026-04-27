"""XGBForecaster: XGBoost-based multi-step time-series forecaster."""

import numpy as np
import xgboost as xgb

from typing import Optional

class XGBForecaster:
    """
    Lightweight wrapper around xgboost.train() for multi-step time-series forecasting.

    Parameters
    ----------
    params : dict, optional
        Hyper-parameters that override the defaults.  ``n_estimators`` and
        ``early_stopping_rounds`` may be included here and are handled
        separately from the tree parameters passed to ``xgb.train``.
        Pass ``{"device": "cpu"}`` when no GPU is available.
    """

    def __init__(
        self,
        params: Optional[dict] = None,
    ) -> None:
        
        # n_estimators and early_stopping_rounds are popped from the params dict in
        # __init__ and stored separately — they are arguments to xgb.train(), not tree parameters.
        params_ = params.copy() # Defensive copy to avoid mutating caller's dict
        self.n_estimators = int(params_.pop("n_estimators"))
        self.early_stopping_rounds = int(params_.pop("early_stopping_rounds"))
        self.params = params_

        self.booster = None # xgb.Booster object created during fit() and used for predict()

    def _make_train_kwargs(self, dtrain: xgb.DMatrix, deval: Optional[xgb.DMatrix]) -> dict:
        """ 
        Helper to construct the kwargs dict for xgb.train() in fit() and cross_validate(). 
        
        Parameters
        ----------
        dtrain : xgb.DMatrix
            Training data.
        deval : xgb.DMatrix, optional
            Validation data for early stopping; if None, early stopping is disabled.

        Returns
        -------
        dict
            Keyword arguments to pass to xgb.train().
        
        """
        d = {"params": self.params, "dtrain": dtrain, "num_boost_round": self.n_estimators}
        if deval is not None:
            d["evals"] = [(deval, "val")]
            d["early_stopping_rounds"] = self.early_stopping_rounds

        return d
    
    def fit(
        self,
        dtrain: xgb.DMatrix,
        deval: Optional[xgb.DMatrix] = None,
    ) -> xgb.Booster:
        """Train the booster.

        Parameters
        ----------
        dtrain : xgb.DMatrix
            Training data. Build with ``make_dmatrix`` before calling this method.
        eval_dm : xgb.DMatrix, optional
            Validation set for early stopping.

        Returns
        -------
        xgb.Booster
            The trained booster (also stored as ``self.booster``).
        """
        train_kwargs = self._make_train_kwargs(dtrain, deval)
        self.booster = xgb.train(**train_kwargs)
        return self.booster

    def predict(self, dm: xgb.DMatrix) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        dm : xgb.DMatrix
            Input data (labels not required).

        Returns
        -------
        np.ndarray of shape (n_samples, n_targets) for multi-output,
        or (n_samples,) for single-output.
        """
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.booster.predict(dm)
