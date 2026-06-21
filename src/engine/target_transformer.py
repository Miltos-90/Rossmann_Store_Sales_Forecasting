"""
This module contains the TargetTransformer class, which is a scikit-learn transformer designed for differencing time series targets.
It allows for stationarizing a time series by calculating the difference between a future target value and a current anchor value,
supporting multi-index DataFrames and Series.
"""

import numpy as np
import pandas as pd

from typing import Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

class TargetTransformer(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer to differencing time series targets.

    This transformer stationarizes a time series by calculating the difference
    between a future target value (defined by a forecast horizon) and a current 
    anchor value. It supports multi-index DataFrames and Series (e.g., grouped 
    by entities alongside a datetime index).

    Attributes:
        forecast_horizon (Any): The frequency string or offset object used to 
            shift the time series (e.g., '1D', '7D', pd.Timedelta).
        anchor_col (str): The column name to use as an anchor when a DataFrame 
            is passed during fitting.
        anchors_ (Optional[Union[pd.Series, pd.DataFrame]]): The historical values 
            stored during `fit` to reverse the differencing during `inverse_transform`.
    """

    def __init__(self, forecast_horizon: pd.DateOffset, anchor_col: str) -> None:
        """Initializes the transformer with a horizon and target anchor columns."""
        self.forecast_horizon = forecast_horizon
        self.anchor_col = anchor_col
        self.anchors_ = None

    def fit(
        self, 
        X: Union[pd.Series, pd.DataFrame], 
        y: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> "TargetTransformer":
        """Extracts and stores anchor values required for inverse transformation.

        Args:
            X (Union[pd.Series, pd.DataFrame]): Features or historical series. If 
                it is a DataFrame, `anchor_col` is extracted.
            y (Optional[Union[pd.Series, pd.DataFrame]]): Target values, used as 
                anchors if X does not match the expected formats.

        Returns:
            TargetTransformer: The fitted transformer instance.
        """
        # If anchors wasn't passed to __init__, grab it from X or y
        if isinstance(X, pd.Series):
            self.anchors_ = X
        elif isinstance(X, pd.DataFrame) and self.anchor_col in X.columns:
            self.anchors_ = X[self.anchor_col]
        elif y is not None:
            self.anchors_ = y
        return self

    def _group_indexes(self, y: pd.Series) -> Tuple[str, list]:
        """Identifies the datetime index and other indexes in a Series.

        Args:
            y (pd.Series): The target time series to analyze.

        Returns:
            tuple: A tuple containing the name of the datetime index and a list 
                of other index names.

        Raises:
            ValueError: If no datetime-like index level is found in `y`.
        """
        datetime_index = None
        other_indexes = []
        
        for level_name in y.index.names:
            index_dtype = y.index.get_level_values(level_name).dtype

            if np.issubdtype(index_dtype, np.datetime64):
                datetime_index = level_name
            else:
                other_indexes.append(level_name)

        if datetime_index is None:
            raise ValueError("No datetime index found in y.")

        return datetime_index, other_indexes


    def _transform_multi_index(self, y: pd.Series, other_indexes: list, n_periods: int) -> pd.Series:
        """ 
        Helper function to transform a multi-index Series by applying the differencing logic to each group defined by the other indexes.

        Args:
            y (pd.Series): The target time series to transform, with a MultiIndex.
            other_indexes (list): A list of index names that are not the datetime index, used for grouping.
            n_periods (int): The number of periods to drop from the start of each group to account for the lagged features.

        Returns:
            pd.Series: The transformed Series with the same MultiIndex, where the differencing has been applied within each group defined by the other indexes.
        """

        y_out = (
                y.reset_index(other_indexes)
                .groupby(other_indexes)
                .apply(lambda x: x.shift(freq=self.forecast_horizon) - x, include_groups=False)
                [y.name]
                .dropna()
        )
        # Drop the first few rows that don't have enough history to compute the lagged features
        y_out = y_out.groupby(other_indexes, group_keys=False).apply(lambda x: x.iloc[n_periods:])
    
        return y_out
    
    def _transform_single_index(self, y: pd.Series, n_periods: int) -> pd.Series:
        """Helper function to transform a single-index Series by applying the differencing logic.

        Args:
            y (pd.Series): The target time series to transform, with a single datetime index.
            n_periods (int): The number of periods to drop from the start to account for the lagged features.

        Returns:
            pd.Series: The transformed Series with the same index, where the differencing has been applied.
        """
        y_out = y.shift(freq=self.forecast_horizon) - y
        y_out = y_out.dropna()

        # Drop the first few rows that don't have enough history to compute the lagged features
        y_out = y_out.iloc[n_periods:]

        return y_out
    def transform(self, y: pd.Series) -> pd.Series:
        """Computes the difference between future values and current values.

        Args:
            y (pd.Series): The target time series to difference. Can contain 
                a MultiIndex.

        Returns:
            pd.Series: The stationarized, differenced time series target with 
                NaN values dropped.

        Raises:
            ValueError: If no datetime-like index level is found in `y`.
        """
        _, other_indexes = self._group_indexes(y)

        # Resolve integer period count for iloc (forecast_horizon is a DateOffset)
        n_periods = abs(next(iter(self.forecast_horizon.kwds.values()), 0))

        # Compute the target as the difference between the future value we want 
        # to predict and the current value, to make the series more stationary 
        # and easier for the model to learn.
        if other_indexes:
            y_out = self._transform_multi_index(y, other_indexes, n_periods)
        else:
            y_out = self._transform_single_index(y, n_periods)

        return y_out

    def inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        """Inverts the differencing and log transformation using index-matched alignment.

        Args:
            y_pred (pd.Series): The predicted differenced values.

        Returns:
            pd.Series: The predictions reverted back to their original target scale, 
                with Sunday values forced to 0.0.

        Raises:
            ValueError: If `fit` has not been called prior to invoking this method.
        """
        if self.anchors_ is None:
            raise ValueError("Fit the transformer first.")

        # Step 1: Inverse differencing via index matching with anchors
        anchors = self.anchors_.loc[y_pred.index]
        y_pred_log = anchors + y_pred

        # Step 2: Inverse log transformation (Note: assumed log1p scale based on expm1)
        y_pred = np.expm1(y_pred_log)

        # Step 3: Zero out Sundays
        
        date_index_name, _ = self._group_indexes(y_pred)  # Fetch date index name to identify the datetime level for Sunday checks
        date_level_num = y_pred.index.names.index(date_index_name)  # Find the level number for the date index to use in set_levels

        # Shift the index according to the forecast horizon to align with the original dates for Sunday checks
        shifted_dates = y_pred.index.levels[date_level_num] - self.forecast_horizon
        y_pred.index  = y_pred.index.set_levels(shifted_dates, level=date_level_num)
        
        is_closed_preds = y_pred.index.get_level_values(date_index_name).day_name() == "Sunday"
        y_pred.loc[is_closed_preds] = 0.0

        return y_pred
