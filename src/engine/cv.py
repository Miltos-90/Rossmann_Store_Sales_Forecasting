""" 
This file contains the implementation of a custom cross-validator for time series data.
"""

import logging
import numpy as np

from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

class TimeSeriesCV(BaseCrossValidator):
    """ 
    TimeSeriesCV is a custom cross-validator for time series data.
    It splits the data into training and testing sets while respecting the temporal order.
    """

    def __init__(self, n_splits: int, train_size: int, test_size: int, gap: int):
        """
        Initialize the TimeSeriesCV cross-validator.

        Args: 
            n_splits (int): Number of splits/folds
            train_size (int): Number of days in the training set
            test_size (int): Number of days in the testing set
            gap (int): Number of days to skip between the training and testing sets to prevent data leakage
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """ 
        Generate indices to split data into training and test set.

        Args:
            X (pd.DataFrame): Feature matrix with a MultiIndex containing 'Store' and 'Date'
            y (pd.Series, optional): Target variable. Not used in this method.
            groups (array-like, optional): Group labels for the samples used while splitting the dataset. Not used in this method.

        Yields:
            train_indices (np.ndarray): Indices of the training set for the current fold
            test_indices (np.ndarray): Indices of the testing set for the current fold
        """

        date_index = X.index.get_level_values("Date")
        unique_dates = date_index.unique().sort_values()
        row_positions = np.arange(len(X))

        for fold in range(1, self.n_splits+1):

            # Test window
            test_end_pos   = len(unique_dates) - 1 - (self.n_splits - fold) * self.test_size
            test_start_pos = test_end_pos - self.test_size + 1

            test_end_date   = unique_dates[test_end_pos]
            test_start_date = unique_dates[test_start_pos]

            # Train window
            train_end_pos   = test_start_pos - self.gap - 1
            train_start_pos = train_end_pos - self.train_size + 1

            train_start_date = unique_dates[train_start_pos]
            train_end_date   = unique_dates[train_end_pos]

            logger.debug(f"Outer Fold {fold}/{self.n_splits} -> "
                         f"Train: {train_start_date.date()} to {train_end_date.date()}, "
                         f"Test: {test_start_date.date()} to {test_end_date.date()}")

            # Get the indices for the outer train and test sets
            train_mask = (date_index >= train_start_date) & (date_index <= train_end_date)
            test_mask  = (date_index >= test_start_date) & (date_index <= test_end_date)

            train_indices = row_positions[train_mask]
            test_indices  = row_positions[test_mask]

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        return self.n_splits
