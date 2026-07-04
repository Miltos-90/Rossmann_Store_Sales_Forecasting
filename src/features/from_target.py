""" 
This module provides functions to generate lagged, differenced, and rolling features for time series data.
The functions are designed to work with pandas Series and DataFrames, allowing for easy integration into time series forecasting workflows.
The generated features can be used to enhance predictive models by providing additional context from historical data.

Functions:

    - make_lags: Create lagged features for a given target series.
    - make_diffs: Create differenced features for a given target series.
    - make_rolling: Create rolling features for a given target series.
"""

import pandas as pd
from typing import Any, Callable, List, Union

def _name_from_offset(offset: pd.DateOffset, prefix: str) -> str:
    """ 
    Generate a name for the lagged feature based on the lag and prefix provided.

    Args:
        offset (pd.DateOffset): The offset for which to generate a name. This should be a pandas DateOffset object.
        prefix (str): A string prefix to prepend to the generated name

    Returns:
        str: A string representing the name of the lagged feature, formatted as '{prefix}_{key}_{value}', where 'key' is the type of lag (e.g., 'days',
    """
    key   = list(offset.kwds.keys())[0]
    value = list(offset.kwds.values())[0]
    return f'{prefix}_{key}_{value}'


def _make_lag(s: pd.Series, lag: pd.DateOffset) -> pd.Series:
    """ 
    Create lagged features for a given target series. This function shifts the target series by the specified lag.
    
    Args:
        s (pd.Series): The target series for which to create lagged features. The index of the series should be a datetime index.
        lag (pd.DateOffset): The lag to apply. This should be a pandas DateOffset object.
    
    Returns:
        pd.Series: A Series containing the lagged features, with the name indicating the lag applied (e.g., 'lag_days_1').
    """
    s_lag = s.shift(freq=lag)
    s_lag.name = _name_from_offset(lag, prefix='lag')
    return s_lag


def _make_rolling(s: pd.Series, window: int, func: str) -> pd.Series:
    """ 
    Create rolling features for a given target series. This function calculates the rolling mean for the target series over specified window sizes.
    
    Args:
        s (pd.Series): The target series for which to create rolling features. The index of the series should be a datetime index.
        window (int): The size of the moving window. This determines how many previous observations are used to calculate the rolling statistic.
        func (str): The aggregation function to apply over the rolling window. This can be any valid pandas aggregation function such as 'mean', 'sum', 'max', 'min', etc.
    
    Returns:
        pd.Series: A Series containing the rolling features, with the name indicating the window size and function applied (e.g., 'rolling_mean_3').
    """

    # Calculate the rolling feature using the specified aggregation function
    rolling_feature = s.rolling(window=window).agg(func)
    rolling_feature.name = f'rolling_{func}_{window}'

    return rolling_feature


def _make_diff(s: pd.Series, diff: pd.DateOffset) -> pd.Series:
    """ 
    Create differenced features for a given target series. This function calculates the difference between the current value and the value at a specified lag.
    
    Args:
        s (pd.Series): The target series for which to create differenced features. The index of the series should be a datetime index.
        diff (pd.DateOffset): The difference to apply. This should be a pandas DateOffset object.
    
    Returns:
        pd.Series: A Series containing the differenced features, with the name indicating the difference applied (e.g., 'diff_days_1').
    """
        
    # Shift the index along the calendar timeline to align historical dates to the target dates
    # Because names might mismatch due to shift, we align on target.index
    s_history = s.shift(freq=diff).reindex(s.index)
    
    # Calculate the difference (Current Target - Historical Target)
    sales_diff = s - s_history

    sales_diff.name = _name_from_offset(diff, prefix='diff')

    return sales_diff


def _generate_features(
    target: pd.Series, 
    items: List[Any], 
    make_func: Callable[..., pd.Series],
    **kwargs: Any
) -> pd.DataFrame:
    """
    Generate features for a given target series using a specified feature creation function.

    Args:
        target (pd.Series): The target series for which to create features. The index of the series should be a datetime index.
        items (List[Any]): A list of items to process.
        make_func (Callable[..., pd.Series]): A function that takes a series and an item from the items list and returns a new series with the generated feature.
        **kwargs (Any): Additional keyword arguments to pass to the make_func.

    Returns:
        pd.DataFrame: A DataFrame containing the generated features, with each column named according to
        the feature creation function and the item processed (e.g., 'lag_days_1', 'diff_days_2', etc.).
    """

    # Ensure chronological order
    processed_series = target.sort_index(ascending=True)

    # Dynamically build the series list using the designated processor function
    features = [make_func(processed_series, item, **kwargs) for item in items]

    # Variable-frequency DateOffsets (months, years) apply calendar arithmetic when
    # shifting the index, which can map multiple distinct source dates to the same
    # target date. For example:
    #   - DateOffset(months=1): both Aug 30 and Aug 31 shift to Sep 30, because
    #     September only has 30 days.
    #   - DateOffset(years=1): both Feb 28 and Feb 29 of a leap year shift to
    #     Feb 28 of the following (non-leap) year.
    # This leaves the shifted Series with duplicate index labels. pd.concat(..., axis=1)
    # then calls .reindex() on each Series to align them to the combined index, and
    # pandas raises ValueError: cannot reindex on an axis with duplicate labels
    # Keeping the last occurrence (the most recent original date that maps to a given
    # shifted date) is the safest default, as it corresponds to the freshest observation
    # available for that calendar slot.
    features = [f[~f.index.duplicated(keep='last')] for f in features]

    # Concatenate and realign to original target index
    result = pd.concat(features, axis=1).reindex(target.index)

    return result


def lag_features(target: pd.Series, lags: list[pd.DateOffset]) -> pd.DataFrame:
    """ 
    Create lagged features for a given target series.

    Args:
        target (pd.Series): The target series for which to create lagged features. The index of the series should be a datetime index.
        lags (list[pd.DateOffset]): A list of pandas DateOffset objects representing the lags to create.

    Returns:
        pd.DataFrame: A DataFrame containing the lagged features, with each column named according to the lag applied (e.g., 'lag_days_1', 'lag_days_2', etc.).
    """
    return _generate_features(target, items=lags, make_func=_make_lag)


def diff_features(target: pd.Series, diffs: list[pd.DateOffset]) -> pd.DataFrame:
    """ 
    Create differenced features for a given target series.
    This function calculates the difference between the current value and the value at a specified lag (y[t]-y[t-lag]).
    During inference y[t] is unknown, so it's best to use lagged  target values to calculate the difference (y[t-lag]-y[t-2*lag]).
    This is why we use the lag parameter to shift the target series before calculating the difference.

    Args:
        target (pd.Series): The target series for which to create differenced features. The index of the series should be a datetime index.
        diffs (list[pd.DateOffset]): A list of pandas DateOffset objects representing the differences to create.

    Returns:
        pd.DataFrame: A DataFrame containing the differenced features, with each column named according to the 
        difference applied (e.g., 'diff_days_1', 'diff_days_2', etc.).
    """
    return _generate_features(target, items=diffs, make_func=_make_diff)


def rolling_features(target: pd.Series, windows: list[int], agg_func: str) -> pd.DataFrame:
    """ 
    Create rolling features for a given target series.
    This function calculates the rolling statistic (e.g., mean, sum, max, min) over a specified window size for the target series.
    During inference y[t] is unknown, so it's best to use lagged target values to calculate the rolling statistic (e.g., rolling_mean(y[t-lag], window)).
    This is why we use the lag parameter to shift the target series before calculating the rolling statistic.

    Args:
        target (pd.Series): The target series for which to create rolling features. The index of the series should be a datetime index.
        windows (list[int]): A list of integers representing the window sizes for the rolling statistic.
        agg_func (str): The aggregation function to apply over the rolling window. This can be any valid pandas aggregation function such as 'mean', 'sum', 'max', 'min', etc.

    Returns:
        pd.DataFrame: A DataFrame containing the rolling features, with each column named according to the window size and function applied (e.g., 'rolling_mean_3', 'rolling_sum_5', etc.).
    """
    return _generate_features(target, items=windows, make_func=_make_rolling, func=agg_func)
