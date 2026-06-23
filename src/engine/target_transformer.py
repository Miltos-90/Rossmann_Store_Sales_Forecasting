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


    def _transform_multi_index(self, y: pd.Series, other_indexes: list) -> pd.Series:
        """ 
        Helper function to transform a multi-index Series by applying the differencing logic to each group defined by the other indexes.

        Args:
            y (pd.Series): The target time series to transform, with a MultiIndex.
            other_indexes (list): A list of index names that are not the datetime index, used for grouping.

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
    
        return y_out
    
    def _transform_single_index(self, y: pd.Series) -> pd.Series:
        """Helper function to transform a single-index Series by applying the differencing logic.

        Args:
            y (pd.Series): The target time series to transform, with a single datetime index.

        Returns:
            pd.Series: The transformed Series with the same index, where the differencing has been applied.
        """
        y_out = y.shift(freq=self.forecast_horizon) - y
        y_out = y_out.dropna()

        return y_out

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Computes the difference between future values and current values.
        It applies the following steps:
        1. Log-transform the target to stabilize variance.
        2. Apply differencing logic.

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

        y_log = np.log1p(y)  # Log-transform the target to stabilize variance

        # Compute the target as the difference between the future value we want 
        # to predict and the current value, to make the series more stationary 
        # and easier for the model to learn.
        if other_indexes:
            y_out = self._transform_multi_index(y_log, other_indexes)
        else:
            y_out = self._transform_single_index(y_log)

        return y_out

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """
        Inverts the differencing and log transformation using index-matched alignment.
        It applies the following steps:
        1. Inverse differencing via index matching with anchors.
        2. Inverse log transformation (assumed log1p scale based on expm1).
        3. Shift the index to get the date of the forecasted values.
        4. Force predictions for Sundays to be 0. 
            -- NOTE: The inverse of this step is not applied in the transform method, 
                     as it is a post-processing step. This is a business rule applied
                     and does not affect the training process.
        5. Round predictions to the nearest integer and convert to int type.

        Args:
            y (pd.Series): The predicted differenced values.

        Returns:
            pd.Series: The predictions reverted back to their original target scale.

        Raises:
            ValueError: If `fit` has not been called prior to invoking this method.
        """
        if self.anchors_ is None:
            raise ValueError("Fit the transformer first.")

        # Step 1: Inverse differencing via index matching with anchors
        anchors = self.anchors_.loc[y.index]
        y_log   = anchors + y

        # Step 2: Inverse log transformation (Note: assumed log1p scale based on expm1)
        y_inv = np.expm1(y_log)

        # Step 3: Shift the index to get the date of the forecasted values
        y_shift = y_inv.copy()
        idx_df  = self._shift_index(y_shift)
        y_shift.index = pd.MultiIndex.from_frame(idx_df)

        # Step 4: Force predictions for Sundays to be 0
        y_shift.loc[self._is_sunday(y_shift)] = 0.0

        # Step 5: Round predictions to the nearest integer and convert to int type
        y_out = y_shift.round(0).astype('int')
        y_out.name = y.name

        return y_out

    def _is_sunday(self, y: pd.Series) -> pd.Series:
        """Checks if the index of the Series corresponds to Sundays.

        Args:
            y (pd.Series): The Series whose index will be checked.

        Returns:
            pd.Series: A boolean Series indicating whether each index corresponds to a Sunday.
        """
        dt_index, _ = self._group_indexes(y)
        is_sunday = y.index.get_level_values(dt_index).day_name() == "Sunday"
        return is_sunday
    
    def _shift_index(self, y: pd.Series) -> pd.DataFrame:
        """Shifts the index of the Series by the forecast horizon.

        Args:
            y (pd.Series): The Series whose index will be shifted.

        Returns:
            pd.DataFrame: A DataFrame representing the shifted index.
        """
        dt_index, _ = self._group_indexes(y)
        idx_df = y.index.to_frame()  # Convert index to a DataFrame
        idx_df[dt_index] = idx_df[dt_index] - self.forecast_horizon  # Shift the specific date column
        return idx_df
