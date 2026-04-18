""" Feature engineering-related functions. """

import pandas as pd
import numpy as np

from typing import Iterable, Dict
from functools import reduce
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


def _to_list(x):
    """
    Normalize input to a list.

    Parameters
    ----------
    x : int or Iterable
        Input value or iterable.

    Returns
    -------
    list
        List containing the input(s).
    """
    return [x] if isinstance(x, (int, DateOffset)) else list(x)


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-format DataFrame to wide format, sorted by index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least three columns.

    Returns
    -------
    pd.DataFrame
        Pivoted and sorted DataFrame.
    """
    return (pd.pivot(df, index=df.columns[0], columns=df.columns[1], values=df.columns[2])
            .sort_index())


def _melt(df_wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Melt a wide-format DataFrame back to long format with a specific value column name.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format DataFrame (pivoted).
    name : str
        Name for the value column.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with MultiIndex.
    """
    index_name = df_wide.index.name
    columns_name = df_wide.columns.name
    return (df_wide.reset_index()
            .melt(id_vars=[index_name], value_name=name)
            .set_index([index_name, columns_name]))


def _align(df: pd.DataFrame, feature_dfs: list) -> pd.DataFrame:
    """
    Merge feature DataFrames with the original DataFrame, preserving row order and columns.

    Parameters
    ----------
    df : pd.DataFrame
        Original long-format DataFrame.
    feature_dfs : list of pd.DataFrame
        List of feature DataFrames to merge.

    Returns
    -------
    pd.DataFrame
        DataFrame with features aligned to the original rows.
    """
    feature_df = pd.concat(feature_dfs, axis=1).reset_index()
    return df.merge(feature_df, how='left').loc[:, feature_df.columns]


def _make_lags(df: pd.DataFrame, lags: int | Iterable[int]) -> pd.DataFrame:
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

    lags = _to_list(lags)
    df_p = _pivot(df)
    lag_dfs = []
    for lag in lags:
        offset = lag if isinstance(lag, DateOffset) else DateOffset(days=lag)
        name = "_".join(f"lag_{v}_{k}" for k, v in offset.kwds.items())
        prior_index = df_p.index - offset
        df_p_lagged = df_p.reindex(prior_index)
        df_p_lagged.index = df_p.index
        lag_dfs.append(_melt(df_p_lagged, name))

    return _align(df, lag_dfs)

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
    lag_df = _make_lags(df, lags)

    return lag_df

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
    - The function relies on a helper function `_make_window_df()` to compute
      each individual rolling mean DataFrame.
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


def _make_cyclic(s: pd.Series, period: int) -> pd.DataFrame:
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

def _make_differences(df: pd.DataFrame,
                      diffs: int | Iterable[int]) -> pd.DataFrame:
    """
    Compute first-order differences of a time series for one or more lag periods,
    and return them in a long-format DataFrame aligned with the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in long format with three columns:
        [date_column, category_column, value_column].
    diffs : int or Iterable[int]
        One or more integer values indicating the number of days over which
        to compute first-order differences (e.g., `[1, 7, 30]` for daily,
        weekly, and monthly changes).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original identifying columns plus new
        difference feature columns named as:
        - `"diff_days_<d>"` for each value of `d` in `diffs`.
        - `"pct_days_<d>"` for each value of `d` in `diffs`.
    """

    diffs = _to_list(diffs)
    df_p = _pivot(df).shift(freq=DateOffset(days=1))  # 1-day lag to prevent data leakage
    feature_dfs = []
    for d in diffs:
        offset = d if isinstance(d, DateOffset) else DateOffset(days=d)
        d_name = "_".join(f"{v}_{k}" for k, v in offset.kwds.items())
        prior_index = df_p.index - offset
        df_p_prior = df_p.reindex(prior_index)
        df_p_prior.index = df_p.index
        feature_dfs.append(_melt(df_p - df_p_prior,                   f"lag_1_{d_name}_diff"))
        feature_dfs.append(_melt((df_p - df_p_prior).div(df_p_prior), f"lag_1_{d_name}_pct_change"))

    return _align(df, feature_dfs)

def make_features(df: pd.DataFrame,
                  lags: int | Iterable[int],
                  roll_windows: Dict[int, int | Iterable[int]],
                  diffs: int | Iterable[int]) -> pd.DataFrame:
    """
    Generate time series forecasting features from a retail sales DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing sales and related metadata per store and date.
    lags : int or Iterable[int]
        List (or single value) of lag periods (in days) to compute past sales values.
        For example, `[1, 7, 30]` creates features for sales 1, 7, and 30 days ago.
    roll_windows : Dict of int and int or Iterable[int]
        Dictionary of rolling window sizes (in days) adn corresponding lags (in days) to compute
        moving averages and other rolling statistics of past sales.
    diffs : Dict of int and int or Iterable[int]
        List (or single value) of differencing periods (in days) for computing
        first-order change features (e.g., daily or weekly sales deltas).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional engineered features, including:
    """

    sales_df = df[['Date', 'Store', 'Sales']]
    
    # Competition-related features
    df['CompetitionDistance'] = df['CompetitionDistance'].apply(np.log1p)
    df['CompetitionSinceMonths'] = ( (df['Date'] - df['CompetitionSinceDate']).dt.days / 30.0 ).round()

    # Promotion-related features
    df['Promo2SinceWeeks'] = (df['Date'] - df['Promo2SinceDate']).dt.days / 7.0
    df['Promo2SinceWeeks'] = df['Promo2SinceWeeks'].fillna(0).round().astype(int) 
    df['Promo2SinceWeeks'] = df['Promo2SinceWeeks'] * df['Promo2'] # essentially a boolean mask

    # Calendar and seasonality features
    df['is_weekend'] = df['Date'].dt.dayofweek >= 5

    # Basic date features
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter

    # Cyclical features
    cyclic_month = _make_cyclic(df['Month'], period=12)
    cyclic_week = _make_cyclic(df['DayOfWeek'], period=7)
    cyclic_month.index = pd.MultiIndex.from_frame(df[['Date', 'Store']])
    cyclic_week.index = pd.MultiIndex.from_frame(df[['Date', 'Store']])

    # Lagged features
    lag_df = _make_lags(sales_df, lags).set_index(['Date', 'Store'])

    # Rolling window features
    window_dfs = []
    for window, lags in roll_windows.items():
        window_df = _make_rolling(sales_df, window, lags).set_index(['Date', 'Store'])
        window_dfs.append(window_df)

    # First-order differencing features
    diff_df = _make_differences(sales_df, diffs).set_index(['Date', 'Store'])

    # Merge everything
    feature_list = [cyclic_month, cyclic_week, lag_df, diff_df] + window_dfs
    merged = reduce(lambda left, right: pd.merge(left, right, on=["Date", "Store"], how="outer"), feature_list).reset_index()
    df = df.merge(merged, how='left')

    # Drop useless columns
    df.drop(['Promo2SinceDate', 'CompetitionSinceDate', 'Sales'], axis=1, inplace=True)

    return df