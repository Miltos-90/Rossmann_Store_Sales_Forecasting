""" Lag features and forecast target generation. """

import pandas as pd

from typing import Iterable
from pandas.tseries.offsets import DateOffset

from .utils import to_list, pivot, melt, align


def make_lags(df: pd.DataFrame, lags: int | Iterable[int]) -> pd.DataFrame:
    """
    Generate and merge multiple lagged feature DataFrames based on one or more DateOffset objects.

    Parameters
    ----------
    df : pd.DataFrame
        Input long-format DataFrame containing at least three columns:
        1. A time or index column (used for pivot index),
        2. A category or variable column (used for pivot columns),
        3. A value column (used for pivot values).
    lags : int or Iterable[int]
        One or more pandas DateOffset objects specifying the temporal lags to apply.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the original columns (apart from the value column)
        plus additional lag feature columns
        (e.g., `"lag_days_7"`, `"lag_weeks_2"`, etc.), merged by date and key.

    Notes
    -----
    - The input DataFrame must be sorted in ascending time order.
    """

    lags = to_list(lags)
    df_p = pivot(df)
    lag_dfs = []
    for lag in lags:
        offset = lag if isinstance(lag, DateOffset) else DateOffset(days=lag)
        name = "_".join(f"lag_{v}_{k}" for k, v in offset.kwds.items())
        prior_index = df_p.index - offset
        df_p_lagged = df_p.reindex(prior_index)
        df_p_lagged.index = df_p.index
        lag_dfs.append(melt(df_p_lagged, name))

    return align(df, lag_dfs)


def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Generate multiple "future-shifted" target columns for each step in the forecast horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Input long-format DataFrame containing at least:
        1. A time or date column,
        2. A category or variable column (if applicable),
        3. A value column representing the target variable.
    horizon : int, default=FORECAST_HORIZON
        The number of future periods (days) to predict.
        For example, `horizon=42` creates targets for `t+1` through `t+42`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing additional columns for each forecast step,
        e.g. `"lag_days_-1"`, `"lag_days_-2"`, … up to the defined horizon.
        Each column represents the target value that occurs that many days ahead.
    """

    lags = range(-1, -(horizon + 1), -1)
    lag_df = make_lags(df, lags)

    lag_df.set_index(['Date', 'Store'], inplace=True)
    return lag_df
