"""XGBForecaster: XGBoost-based multi-step time-series forecaster."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from typing import Optional

class _BatchIter(xgb.DataIter):
    """Iterate over (X, y) in fixed-size row batches for use with QuantileDMatrix."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
        self._X = X
        self._y = y
        self._batch_size = batch_size
        self._it = 0
        super().__init__()  # no cache_prefix — data stays in memory

    def next(self, input_data) -> int:
        start = self._it * self._batch_size
        if start >= len(self._X):
            return 0
        end = min(start + self._batch_size, len(self._X))
        input_data(data=self._X[start:end], label=self._y[start:end])
        self._it += 1
        return 1

    def reset(self) -> None:
        self._it = 0


class XGBForecaster:
    """XGBoost-based multi-step time-series forecaster.

    Uses the native ``xgb.train`` / ``xgb.Booster`` API with ``xgb.DMatrix``
    inputs for maximum performance.

    Parameters
    ----------
    horizon : int
        Number of future periods to forecast simultaneously (width of y).
    params : dict, optional
        Hyper-parameters that override the defaults.  ``n_estimators`` and
        ``early_stopping_rounds`` may be included here and are handled
        separately from the tree parameters passed to ``xgb.train``.
        Pass ``{"device": "cpu"}`` when no GPU is available.
    batch_size : int or None, optional
        When set, training data is fed to XGBoost in batches of this many
        rows via ``QuantileDMatrix`` + ``DataIter``.  This keeps GPU memory
        usage bounded at the cost of extracting the raw arrays from the
        input ``DMatrix``.  ``None`` (default) passes the ``DMatrix``
        directly to ``xgb.train`` without batching.
    """

    def __init__(
        self,
        horizon: int,
        params: Optional[dict] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        
        self.horizon = horizon
        self.batch_size = batch_size

        # n_estimators and early_stopping_rounds are popped from the params dict in
        # __init__ and stored separately — they are arguments to xgb.train(), not tree parameters.
        params_ = params.copy() # Defensive copy to avoid mutating caller's dict
        self.n_estimators: int = int(params_.pop("n_estimators"))
        self.early_stopping_rounds: int = int(params_.pop("early_stopping_rounds"))
        self.params: dict = params_

        self.booster = None # xgb.Booster object created during fit() and used for predict()

    def fit(
        self,
        dtrain: xgb.DMatrix,
        eval_dm: Optional[xgb.DMatrix] = None,
    ) -> xgb.Booster:
        """Train the booster.

        Parameters
        ----------
        dtrain : xgb.DMatrix
            Training data with labels embedded (created via
            ``xgb.DMatrix(X, label=y)``).
        eval_dm : xgb.DMatrix, optional
            Validation set for early stopping.

        Returns
        -------
        xgb.Booster
            The trained booster (also stored as ``self.booster``).
        """
        if self.batch_size is not None:
            # Extract raw arrays from the DMatrix and
            # rebuild as QuantileDMatrix fed through a DataIterator
            # for batched training.
            
            n_rows = dtrain.num_row()
            X_arr = dtrain.get_data().toarray()          # CSR → dense numpy
            y_flat = dtrain.get_label()                  # always 1-D
            y_arr = y_flat.reshape(n_rows, self.horizon) if self.horizon > 1 else y_flat
            dtrain_used = xgb.QuantileDMatrix(_BatchIter(X_arr, y_arr, self.batch_size))
        
        else:
            dtrain_used = dtrain

        train_kwargs: dict = {
            "params": self.params,
            "dtrain": dtrain_used,
            "num_boost_round": self.n_estimators,
        }
        if eval_dm is not None:
            train_kwargs["evals"] = [(eval_dm, "val")]
            train_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

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
        np.ndarray of shape (n_samples, horizon) for multi-output,
        or (n_samples,) for single-output.
        """
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.booster.predict(dm)
