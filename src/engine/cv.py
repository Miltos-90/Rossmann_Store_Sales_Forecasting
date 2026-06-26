"""Time-series cross-validator for (Date, Store) MultiIndex DataFrames."""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Iterator
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

class TimeSeriesCV(BaseCrossValidator):
    """Sliding-window time-series cross-validator.

    Splits are defined on the *Date* level of a ``(Date, Store)`` MultiIndex
    so that every store is present in both the train and test set of each fold.

    Each fold has a fixed ``train_size``-day training window followed by a
    ``test_size``-day test window, optionally separated by a ``gap``-day
    embargo.  The windows slide forward in non-overlapping steps of
    ``test_size`` days.  Fold 1 is the earliest and fold ``n_splits`` is the
    latest (anchored to the end of the dataset).

    The ``gap`` embargo drops the last ``gap`` dates immediately before each
    test window from training.  This prevents leakage when the target is
    forward-looking (e.g. a difference or value over a forecast horizon):
    without a gap, the last ``horizon`` training rows would carry labels
    built from observations that fall inside the test window.  Set
    ``gap = forecast_horizon`` to remove that overlap.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    train_size : int
        Number of days in each training window per store.
    test_size : int
        Number of days in each test window per store.  Also the step between consecutive
        folds.
    gap : int, default 0
        Number of days to drop between the end of the training window and the
        start of the test window (embargo).  Use the forecast horizon to avoid
        forward-looking target leakage.

    Examples
    --------
    >>> cv = TimeSeriesCV(n_splits=3, train_size=180, test_size=7, gap=7)
    >>> for train_idx, test_idx in cv.split(X):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int,
        train_size: int,
        test_size: int,
        gap: int = 0,
    ) -> None:
        """ Initialize the cross-validator. """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {train_size}")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        self.n_splits   = n_splits
        self.train_size = train_size
        self.test_size  = test_size
        self.gap        = gap


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
        date_values = X.index.get_level_values("Date")
        row_positions = np.arange(len(X))

        for fold in range(1, self.n_splits + 1):
            # Test window: fold 1 is earliest, fold n_splits is latest (ends at last date)
            test_end_pos   = n_dates - 1 - (self.n_splits - fold) * self.test_size
            test_start_pos = test_end_pos - self.test_size + 1

            test_start_date = dates[test_start_pos]
            test_end_date   = dates[test_end_pos]

            # Train window: fixed train_size days ending `gap` days before the test
            # window. The gap embargoes the last `gap` dates so forward-looking
            # targets cannot peek into the test set.
            train_end_pos   = test_start_pos - 1 - self.gap
            train_start_pos = train_end_pos - self.train_size + 1

            if train_start_pos < 0:
                raise ValueError(
                    f"Not enough dates for fold {fold}/{self.n_splits}: need "
                    f"train_size={self.train_size} + gap={self.gap} + "
                    f"{self.n_splits} x test_size={self.test_size} dates, but only "
                    f"{n_dates} unique dates are available."
                )

            train_start_date = dates[train_start_pos]
            train_end_date   = dates[train_end_pos]

            train_mask = (date_values >= train_start_date) & (date_values <= train_end_date)
            test_mask  = (date_values >= test_start_date)  & (date_values <= test_end_date)

            train_idx = row_positions[train_mask]
            test_idx  = row_positions[test_mask]

            logger.info(
                f"Fold {fold}/{self.n_splits}  train={len(train_idx)} samples [{train_start_date.date()}" 
                f" -> {train_end_date.date()}]  test={len(test_idx)} samples [{test_start_date.date()} -> "
                f"{test_end_date.date()}]"
            )

            yield train_idx, test_idx


    def get_n_splits(self) -> int:
        return self.n_splits
