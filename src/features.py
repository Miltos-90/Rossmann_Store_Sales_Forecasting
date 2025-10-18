""" Feature engineering-related functions. """

import pandas as pd
import numpy as np

from typing import Iterable
from pandas.tseries.offsets import DateOffset

def _in_promo2(row, date_col: str, interval_col: str, start_promo_date_col: str):
    """
    Determine whether a given observation falls within an active Promo2 period.

    Parameters
    ----------
    row : pd.Series
        A single row from a DataFrame, typically passed by `DataFrame.apply(axis=1)`.
    date_col : str
        The column name in `row` containing the date to check.
    interval_col : str
        The column name containing the active Promo2 intervals as strings,
        typically month abbreviations (e.g., "Feb,May,Aug,Nov").
    start_promo_date_col : str
        The column name containing the start date of the Promo2 campaign for the store.

    Returns
    -------
    bool
        True if the store is in an active Promo2 period for the given date, False otherwise.
    """

    month = row[date_col].strftime("%b")

    if pd.isna(month) or pd.isna(row[interval_col]):
        out = False
    else:
        out = (row[date_col] >= row[start_promo_date_col]) & (month in row[interval_col])

    return out

def attach_store_data(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge store-level metadata into the main DataFrame and compute active Promo2 flags.

    This function joins transactional or daily sales data (`df`) with store metadata (`stores`)
    on the `'Store'` column. After merging, it determines whether each observation falls within
    an active Promo2 period using the `_in_promo2()` helper function, encoding the result as an
    integer indicator (1 for active, 0 for inactive). The `PromoInterval` column is dropped
    after processing to prevent duplication.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame containing daily or transactional data.
        Must include at least the following columns:
        - `'Store'`: store identifier,
        - `'Date'`: observation date.
    stores : pd.DataFrame
        Store metadata DataFrame containing additional store-level attributes.
        Must include at least:
        - `'Store'`: store identifier (to join on),
        - `'PromoInterval'`: string listing active promo months (e.g., "Feb,May,Aug,Nov"),
        - `'Promo2SinceDate'`: datetime marking when Promo2 started for the store.

    Returns
    -------
    pd.DataFrame
        The input `df` enriched with store-level attributes and a new column:
        - `'Promo2'`: integer flag (1 if active Promo2, 0 otherwise).
    """

    df = df.merge(stores, on='Store')
    df['Promo2'] = df.apply(_in_promo2, args=('Date', 'PromoInterval', 'Promo2SinceDate'), axis=1).astype(int)
    df.drop('PromoInterval', axis=1, inplace=True)

    return df

def _make_lag_df(df: pd.DataFrame, lag: DateOffset) -> pd.DataFrame:
    """
    Create a lagged version of a time-indexed DataFrame, reshaped for feature generation.

    This function shifts the input DataFrame by a specified pandas `DateOffset` (e.g., `pd.DateOffset(days=7)`),
    then melts the result into a long format suitable for merging or stacking multiple lagged features.
    The lag offset parameters are encoded into the column name of the lagged values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex (or time-based index) and one or more columns of values.
    lag : pd.DateOffset
        The pandas DateOffset object defining the temporal lag to apply
        (e.g., `pd.DateOffset(days=7)`, `pd.DateOffset(weeks=1)`).

    Returns
    -------
    pd.DataFrame
        A melted DataFrame containing the lagged values, with a multi-index consisting of
        the original index name and column name, and columns:
        - The lagged value column named like `"lag_<unit>_<amount>"` (e.g., `"lag_days_7"`).
    """

    id_name = df.index.name
    value_name = "_".join(f"lag_{k}_{v}" for k, v in lag.kwds.items())
    melt_index = [df.index.name, df.columns.name]

    lag_df = (df.shift(freq=lag)
              .reset_index()
              .melt(id_vars=[id_name], value_name=value_name)
              .set_index(melt_index))

    return lag_df

def make_lags(df: pd.DataFrame, lags: DateOffset | Iterable[DateOffset]) -> pd.DataFrame:
    """
    Generate and merge multiple lagged feature DataFrames based on one or more DateOffset objects.

    Parameters
    ----------
    df : pd.DataFrame
        Input long-format DataFrame containing at least three columns:
        1. A time or index column (used for pivot index),
        2. A category or variable column (used for pivot columns),
        3. A value column (used for pivot values).
    lags : pd.DateOffset or Iterable[pd.DateOffset]
        One or more pandas DateOffset objects specifying the temporal lags to apply,
        e.g. `pd.DateOffset(days=7)` or `[pd.DateOffset(days=7), pd.DateOffset(weeks=2)]`.

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

    if isinstance(lags, DateOffset):
        lags = [lags]

    df_pivot = (pd.pivot(df,
                         index=df.columns[0],
                         columns=df.columns[1],
                         values=df.columns[2])
                .sort_index()) # Sorted from oldest to newest

    lag_dfs = [_make_lag_df(df_pivot, lag) for lag in lags]
    lag_df = pd.concat(lag_dfs, axis=1).reset_index()

    # we need to merge with the input dataframe to keep only the dates
    # that appear on the input dataframe.
    lag_df_merged = df.merge(lag_df, how='left').loc[:, lag_df.columns]

    return lag_df_merged

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

    Notes
    -----
    - This function internally constructs negative `DateOffset` objects to shift values forward in time.
    """
        
    lags = range(-1, -(horizon + 1), -1)
    date_lags = [DateOffset(days=lag) for lag in lags]
    lag_df = make_lags(df, date_lags)

    return lag_df

def _make_window_df(df: pd.DataFrame, window: int, lag: int) -> pd.DataFrame:
    """
    Create a long-format DataFrame containing rolling mean values over a specified window and lag.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time series or sequential data. The DataFrame's index
        should represent the time or sequence dimension.
    window : int
        The size of the rolling window (in number of observations).
    lag : int
        A lag identifier used only for naming the resulting column; it does not affect
        computation directly.

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with the rolling mean values. The index will be a
        MultiIndex composed of the original index name and the column name, and the
        resulting column will be named
        ``rolling_mean_days_{window}_lag_{lag}``.

    """

    melt_index = [df.index.name, df.columns.name]
    id_name = df.index.name

    wdf = (df
           .rolling(window=window).mean()
           .reset_index()
           .melt(id_vars=[id_name], value_name=f"rolling_mean_days_{window}_lag_{lag}")
           .set_index(melt_index))
    
    return wdf

def make_rolling(df: pd.DataFrame, windows: int | Iterable[int], lag: int = 1) -> pd.DataFrame:
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
    lag : int
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
    - The function relies on a helper function `_make_window_df()` to compute
      each individual rolling mean DataFrame.
    - The index is expected to represent ordered dates or times; sorting is
      applied to ensure chronological order.
    - The lag operation shifts the time index forward by the specified number
      of days.
    """

    if not isinstance(windows, list):
        windows = [windows]

    df_pivot = (pd.pivot(df,
                        index=df.columns[0],
                        columns=df.columns[1],
                        values=df.columns[2])
                        .sort_index()  # Sorted from oldest to newest
                        .shift(freq=DateOffset(days=lag)))  # single-day-lag

    window_dfs = [_make_window_df(df_pivot, window=w, lag=lag) for w in windows]
    window_df = pd.concat(window_dfs, axis=1).reset_index()

    # we need to merge with the input dataframe to keep only the dates
    # that appear on the input dataframe.
    window_df_merged = df.merge(window_df, how='left').loc[:, window_df.columns]

    return window_df_merged

def make_cyclic(s: pd.Series, period: int) -> pd.DataFrame:
    """
    Convert a periodic numeric pandas Series into its cyclic (sin and cos) representation.

    Parameters
    ----------
    s : pd.Series
        Input pandas Series containing numeric values representing a cyclic variable 
        (e.g., month numbers, hours of day, days of week).
    period : int
        The period of the cycle (e.g., 24 for hours, 7 for days of week, 12 for months).

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - `<s.name>_sin`: sine transformation of the input series.
        - `<s.name>_cos`: cosine transformation of the input series.
    """

    cyclic_df = pd.concat([np.sin(2 * np.pi * s / period),
                        np.cos(2 * np.pi * s / period)],
                        axis=1)

    cyclic_df.columns=[f'{s.name}_sin', f'{s.name}_cos']

    return cyclic_df