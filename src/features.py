""" Feature engineering-related functions. """

import pandas as pd
import numpy as np

from typing import Iterable
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

    if isinstance(lags, int):
        lags = [lags]

    df_p = (
        pd.pivot(df, index=df.columns[0], columns=df.columns[1], values=df.columns[2])
        .sort_index() # Sorted from oldest to newest
    )

    id_name = df_p.index.name
    melt_index = [df_p.index.name, df_p.columns.name]
    lag_dfs = []
    for lag in lags:
        lag_offset = DateOffset(days=lag)
        value_name = "_".join(f"lag_{v}_{k}" for k, v in lag_offset.kwds.items())

        lag_df = (df_p.shift(freq=lag_offset)
                .reset_index()
                .melt(id_vars=[id_name], value_name=value_name)
                .set_index(melt_index))

        lag_dfs.append(lag_df)

    # we need to merge with the input dataframe to keep only the dates
    # that appear on the input dataframe and preserve row order.
    lag_df = pd.concat(lag_dfs, axis=1).reset_index()
    lag_df = df.merge(lag_df, how='left').loc[:, lag_df.columns]

    return lag_df

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

    if not isinstance(windows, list):
        windows = [windows]

    if not isinstance(lags, list):
        lags = [lags]

    df_p = (pd.pivot(df,
                    index=df.columns[0],
                    columns=df.columns[1],
                    values=df.columns[2])
            .sort_index())  # Sorted from oldest to newest

    melt = lambda df, name: (df.reset_index()
                            .melt(id_vars=[df_p.index.name], value_name=name)
                            .set_index([df_p.index.name, df_p.columns.name]))

    feature_dfs = []
    for lag in lags:
        df_p_lagged = df_p.shift(freq=DateOffset(days=lag))

        for w in windows:
            roll = df_p_lagged.rolling(window=w)
            feature_list = pd.concat([
                melt(roll.mean(), name=f"lag_{lag}_roll_{w}_days_mean"),
                melt(roll.std(), name=f"lag_{lag}_roll_{w}_days_std"),
                melt(roll.skew(), name=f"lag_{lag}_roll_{w}_days_skew"),
                melt(roll.kurt(), name=f"lag_{lag}_roll_{w}_days_kurt"),
                melt(roll.quantile(0.5), name=f"lag_{lag}_roll_{w}_days_median"),
                melt(roll.quantile(0.1), name=f"lag_{lag}_roll_{w}_days_10percentile"),
                melt(roll.quantile(0.9), name=f"lag_{lag}_roll_{w}_days_90percentile")
            ], axis=1)

            feature_dfs.append(feature_list)

    # we need to merge with the input dataframe to keep only the dates
    # that appear on the input dataframe and preserve row order.
    feature_df = pd.concat(feature_dfs, axis=1).reset_index()
    feature_df = df.merge(feature_df, how='left').loc[:, feature_df.columns]

    return feature_df

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

    if not isinstance(diffs, list):
        diffs = [diffs]

    df_p = (
        pd.pivot(df, index=df.columns[0], columns=df.columns[1], values=df.columns[2])
        .sort_index()  # Sorted from oldest to newest
        .shift(freq=DateOffset(days=1))  # single-day-lag. We compute differences of the lagged value
        )

    id_name = df_p.index.name
    melt_index = [df_p.index.name, df_p.columns.name]
    melt = lambda df, name: (df.reset_index()
                             .melt(id_vars=[id_name], value_name=name)
                             .set_index(melt_index))

    diff_list = [melt(df_p.diff(d), name=f"lag_1_{d}_days_diff")
                 for d in diffs]
    pct_list = [melt(df_p.pct_change(d, fill_method=None), name=f"lag_1_{d}_days_pct_change")
                for d in diffs]
    feature_list = diff_list + pct_list

    # we need to merge with the input dataframe to keep only the dates
    # that appear on the input dataframe and preserve row order.
    feature_df = pd.concat(feature_list, axis=1).reset_index()
    feature_df = df.merge(feature_df, how='left').loc[:, feature_df.columns]

    return feature_df

from typing import Dict

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