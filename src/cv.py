"""Time-series cross-validator for (Date, Store) MultiIndex DataFrames."""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Iterator
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

class TimeSeriesCV(BaseCrossValidator):
    """Sliding/expanding-window time-series cross-validator.

    Splits are defined on the *Date* level of a ``(Date, Store)`` MultiIndex
    so that every store is present in both the train and test set of each fold.

    The last ``n_splits * horizon`` dates of the dataset are reserved as test
    windows.  Each fold's test set is one ``horizon``-length block, ordered
    chronologically (fold 1 is the earliest, fold ``n_splits`` the latest).
    The corresponding train set is the ``train_size`` days immediately
    preceding that test window.  If ``train_size`` is ``None`` all available
    data before the test window is used (expanding window).

    Parameters
    ----------
    n_splits : int
        Number of folds.  Determines how many horizon-length blocks are taken
        from the end of the dataset as test sets.
    horizon : int
        Length of each test set in days.
    train_size : int or None
        Number of days in the training set.  Uses the most-recent days that
        fall before the test window.  ``None`` means use all prior data.

    Examples
    --------
    >>> cv = TimeSeriesCV(n_splits=3, horizon=7, train_size=180)
    >>> for train_idx, test_idx in cv.split(X):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int,
        horizon: int,
        train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if train_size is not None and train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {train_size}")
        self.n_splits   = n_splits
        self.horizon    = horizon
        self.train_size = train_size


    def split(
        self, X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` positional index arrays.

        Parameters
        ----------
        X : pd.DataFrame
            Must have a ``(Date, Store)`` MultiIndex.

        Yields
        ------
        train_idx : np.ndarray of int
        test_idx  : np.ndarray of int
        """
        dates = (
            X.index.get_level_values("Date")
            .unique()
            .sort_values()
        )
        n_dates = len(dates)
        total_test_days = self.n_splits * self.horizon

        if total_test_days >= n_dates:
            raise ValueError(
                f"n_splits * horizon = {total_test_days} >= n_dates = {n_dates}. "
                "Not enough data for training. Reduce n_splits or horizon."
            )

        date_values   = X.index.get_level_values("Date")
        row_positions = np.arange(len(X))

        for fold in range(1, self.n_splits + 1):
            # Test window: fold 1 is the earliest, fold n_splits the latest
            test_end_pos   = n_dates - (self.n_splits - fold) * self.horizon - 1
            test_start_pos = test_end_pos - self.horizon + 1

            test_start_date = dates[test_start_pos]
            test_end_date   = dates[test_end_pos]

            # Train window: directly before the test window
            train_end_pos = test_start_pos - 1
            if self.train_size is None:
                train_start_pos = 0
            else:
                train_start_pos = max(0, train_end_pos - self.train_size + 1)

            train_start_date = dates[train_start_pos]
            train_end_date   = dates[train_end_pos]

            train_mask = (date_values >= train_start_date) & (date_values <= train_end_date)
            test_mask  = (date_values >= test_start_date)  & (date_values <= test_end_date)

            train_idx = row_positions[train_mask]
            test_idx  = row_positions[test_mask]

            self._check_stores(X, train_idx, test_idx, fold)

            logger.debug(
                "Fold %d/%d  train=%d samples [%s → %s]  test=%d samples [%s → %s]",
                fold, self.n_splits,
                len(train_idx), train_start_date.date(), train_end_date.date(),
                len(test_idx),  test_start_date.date(),  test_end_date.date(),
            )

            yield train_idx, test_idx


    def get_n_splits(self) -> int:
        return self.n_splits


    def _check_stores(
        self,
        X: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold: int,
    ) -> None:
        """Warn if any store is missing from one side of the split."""
        train_stores  = set(X.iloc[train_idx].index.get_level_values("Store"))
        test_stores   = set(X.iloc[test_idx].index.get_level_values("Store"))
        missing_train = test_stores - train_stores
        missing_test  = train_stores - test_stores
        if missing_train:
            warnings.warn(
                f"Fold {fold}: stores {missing_train} appear in test but not train.",
                stacklevel=3,
            )
        if missing_test:
            warnings.warn(
                f"Fold {fold}: stores {missing_test} appear in train but not test.",
                stacklevel=3,
            )
