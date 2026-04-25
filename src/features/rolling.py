""" Rolling-window statistics features (mean, std, skew, kurt, median, percentiles). """

import pandas as pd

from typing import Iterable
from pandas.tseries.offsets import DateOffset

from .utils import _to_list, _pivot, _melt, _align


def _make_rolling(df: pd.DataFrame,
                  windows: int | Iterable[int],
                  lags: int | Iterable[int]) -> pd.DataFrame:
    """
    Compute rolling mean features for one or more window sizes, with an optional lag shift.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame in long format, with at least three columns:
        - The first column is used as the index (typically a date or time field).
        - The second column represents variable names (categories or series).
        - The third column contains the numeric values to aggregate.
    windows : int or iterable of int
        Rolling window size(s) in number of observations (e.g., 3, 7, or [3, 7, 14]).
        If a single integer is passed, it is automatically wrapped in a list.
    lags : int or iterable of int
        Number of days (or time units) to lag the data before computing rolling
        statistics.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original date/variable/value columns along
        with new rolling mean columns for each specified window and lag. Each
        rolling column is named according to the pattern:
        ``rolling_mean_days_{window}_lag_{lag}``.

    Notes
    -----
    - The index is expected to represent ordered dates or times; sorting is
      applied to ensure chronological order.
    - The lag operation shifts the time index forward by the specified number
      of days.
    """

    windows = _to_list(windows)
    lags = _to_list(lags)
    df_p = _pivot(df)
    feature_dfs = []
    for lag in lags:
        offset = lag if isinstance(lag, DateOffset) else DateOffset(days=lag)
        lag_name = "_".join(f"{v}_{k}" for k, v in offset.kwds.items())
        prior_index = df_p.index - offset
        df_p_lagged = df_p.reindex(prior_index)
        df_p_lagged.index = df_p.index
        for w in windows:
            roll = df_p_lagged.rolling(window=w)
            feature_dfs.extend([
                _melt(roll.mean(),         f"lag_{lag_name}_roll_{w}_days_mean"),
                _melt(roll.std(),          f"lag_{lag_name}_roll_{w}_days_std"),
                _melt(roll.skew(),         f"lag_{lag_name}_roll_{w}_days_skew"),
                _melt(roll.kurt(),         f"lag_{lag_name}_roll_{w}_days_kurt"),
                _melt(roll.quantile(0.5),  f"lag_{lag_name}_roll_{w}_days_median"),
                _melt(roll.quantile(0.1),  f"lag_{lag_name}_roll_{w}_days_10percentile"),
                _melt(roll.quantile(0.9),  f"lag_{lag_name}_roll_{w}_days_90percentile"),
            ])

    return _align(df, feature_dfs)
