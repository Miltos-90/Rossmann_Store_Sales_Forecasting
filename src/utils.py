"""Utility functions for XGBoost data handling."""

import numpy as np
import xgboost as xgb

from typing import Optional

class _BatchIter(xgb.DataIter):
    """Iterate over (X, y) in fixed-size row batches for use with QuantileDMatrix."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
        """Initialize the batch iterator with the full dataset and batch size."""
        self._X = X
        self._y = y
        self._batch_size = batch_size
        self._it = 0
        super().__init__()  # no cache_prefix — data stays in memory

    def next(self, input_data: xgb.DataIter) -> int:
        """Load the next batch into the provided input_data object."""
        start = self._it * self._batch_size
        if start >= len(self._X):
            return 0
        end = min(start + self._batch_size, len(self._X))
        input_data(data=self._X[start:end], label=self._y[start:end])
        self._it += 1
        return 1

    def reset(self) -> None:
        """Reset the iterator to the beginning of the dataset."""
        self._it = 0


def make_dmatrix(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
) -> xgb.DMatrix:
    """Build a DMatrix or QuantileDMatrix from numpy arrays.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray, optional
        Labels. Pass ``None`` when building an inference-only matrix.
    batch_size : int, optional
        When set, returns a QuantileDMatrix fed via DataIter in chunks of
        this many rows. Use for GPU training to keep memory bounded.

    Returns
    -------
    xgb.DMatrix or xgb.QuantileDMatrix
    """
    if batch_size is not None:
        return xgb.QuantileDMatrix(_BatchIter(X, y, batch_size))
    return xgb.DMatrix(X, label=y)