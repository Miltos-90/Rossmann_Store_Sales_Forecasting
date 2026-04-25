"""
Feature engineering for the Rossmann Store Sales forecasting pipeline.

All features are constructed from ``Shifted_Sales`` (i.e. ``Sales`` shifted
forward by one day per store) to prevent any leakage of current-day sales
into the feature set.  The ``lags``, ``diffs``, and ``roll_windows`` arguments
passed to ``make_features`` are expressed in terms of the **original Sales
axis**; the function internally reduces every offset by one day to compensate
for the shift.

Features produced by ``make_features``
---------------------------------------

Competition
~~~~~~~~~~~
- ``CompetitionDistance``       : log1p-transformed distance (metres) to the nearest competitor.
- ``CompetitionSinceMonths``    : number of months elapsed since the nearest competitor opened.

Calendar / seasonality
~~~~~~~~~~~~~~~~~~~~~~
- ``Year``                      : calendar year.
- ``Month``                     : calendar month (1–12).
- ``Quarter``                   : calendar quarter (1–4).
- ``DayOfWeek``                 : day of week (1 = Monday … 7 = Sunday).
- ``is_weekend``                : bool – True for Saturday and Sunday.
- ``is_month_start``            : bool – True for days 1–3 of the month.
- ``is_month_end``              : bool – True for days 28–31 of the month.
- ``DayOfMonth_sin / _cos``     : cyclic (sin/cos) encoding of the day of month (period 31).
- ``WeekOfYear_sin / _cos``     : cyclic (sin/cos) encoding of the ISO week number (period 52).

Promotion
~~~~~~~~~
- ``Promo``                     : binary flag for the regular one-time promotion.
- ``Promo2``                    : binary flag indicating an active running Promo2 interval
                                  for the store on that date.
- ``consecutive_promo_days``    : number of consecutive days the store has been in a
                                  Promo streak up to and including the current day.
- ``consecutive_promo2_days``   : same for Promo2.

School / state holidays
~~~~~~~~~~~~~~~~~~~~~~~
- ``SchoolHoliday``             : binary flag for a school holiday.
- ``StateHoliday``              : categorical code for public holidays (0 = none, a/b/c = type).
- ``days_to_next_state_holiday``: days until the next public holiday for the store's state.
- ``days_since_last_state_holiday``: days since the most recent public holiday.
- ``days_to_next_school_holiday``: days until the next school holiday.

Lag features  (one column per lag in ``lags``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``lag_<n>_<unit>`` where n/unit are derived from the adjusted DateOffset
(e.g. ``lag_6_days`` corresponds to a 7-day lag on original Sales).
Each value is ``Shifted_Sales`` looked up ``n <unit>`` before the current date.

Rolling-window statistics  (per window size × lag combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``lag_<lag>_roll_<w>_days_<stat>`` where ``<stat>`` ∈
{mean, std, skew, kurt, median, 10percentile, 90percentile}.
Computed over a window of ``w`` trading days on ``Shifted_Sales`` starting
from the adjusted lag offset.

First-order differences  (one pair per entry in ``diffs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``lag_1_<n>_<unit>_diff``        : absolute change in ``Shifted_Sales`` over the period.
- ``lag_1_<n>_<unit>_pct_change``  : relative (%) change over the same period.

Target-encoded categoricals  (time-aware, per store)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``<col>_te``.  For each categorical column listed below, the value is
the expanding historical mean of ``Shifted_Sales`` for the (store, category)
pair up to — but **not including** — the current date, preventing lookahead
leakage.

Encoded columns: ``Store``, ``Promo``, ``Promo2``, ``SchoolHoliday``,
``Assortment``, ``StoreType``, ``StateHoliday``, ``DayOfWeek``,
``Quarter``, ``Year``, ``Month``.
"""

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
    df_p = _pivot(df)
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

def _make_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proximity in days to the nearest state and school holidays per store.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'Store', 'StateHoliday', 'SchoolHoliday' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', 'Store', 'days_to_next_state_holiday',
        'days_since_last_state_holiday', 'days_to_next_school_holiday'].
    """
    ns_per_day = 86_400 * 10 ** 9  # nanoseconds in a day; used to convert int64 timestamp differences back to days
    result_dfs = []

    for store_id, group in df[['Date', 'Store', 'StateHoliday', 'SchoolHoliday']].groupby('Store'):
        group = group.sort_values('Date')
        # Convert dates to int64 nanoseconds so we can do fast arithmetic with numpy
        dates_ns = group['Date'].values.astype('int64')

        # StateHoliday is '0' (string) or 0 (int) when there is no holiday; anything else is a real holiday
        is_state_hol = ~group['StateHoliday'].isin(['0', 0]) & group['StateHoliday'].notna()
        state_hol_ns = group.loc[is_state_hol, 'Date'].values.astype('int64')  # sorted holiday timestamps

        is_school_hol = group['SchoolHoliday'] == 1
        school_hol_ns = group.loc[is_school_hol, 'Date'].values.astype('int64')  # sorted holiday timestamps

        if len(state_hol_ns) > 0:
            # searchsorted(side='left') gives the index of the first holiday >= current date
            idx_next = np.searchsorted(state_hol_ns, dates_ns, side='left')
            has_next = idx_next < len(state_hol_ns)  # False for dates after the last known holiday
            days_to_next_state = np.where(
                has_next,
                (state_hol_ns[np.minimum(idx_next, len(state_hol_ns) - 1)] - dates_ns) / ns_per_day,
                np.nan)  # NaN when no future holiday exists in the dataset

            # searchsorted(side='right') - 1 gives the index of the last holiday <= current date
            idx_prev = np.searchsorted(state_hol_ns, dates_ns, side='right') - 1
            has_prev = idx_prev >= 0  # False for dates before the first known holiday
            days_since_last_state = np.where(
                has_prev,
                (dates_ns - state_hol_ns[np.maximum(idx_prev, 0)]) / ns_per_day,
                np.nan)  # NaN when no past holiday exists in the dataset
        else:
            days_to_next_state = np.full(len(group), np.nan)
            days_since_last_state = np.full(len(group), np.nan)

        if len(school_hol_ns) > 0:
            idx_next = np.searchsorted(school_hol_ns, dates_ns, side='left')
            has_next = idx_next < len(school_hol_ns)
            days_to_next_school = np.where(
                has_next,
                (school_hol_ns[np.minimum(idx_next, len(school_hol_ns) - 1)] - dates_ns) / ns_per_day,
                np.nan)
        else:
            days_to_next_school = np.full(len(group), np.nan)

        result_dfs.append(pd.DataFrame({
            'Date': group['Date'].values,
            'Store': store_id,
            'days_to_next_state_holiday': days_to_next_state,
            'days_since_last_state_holiday': days_since_last_state,
            'days_to_next_school_holiday': days_to_next_school,
        }))

    return pd.concat(result_dfs, ignore_index=True)


def _make_consecutive_promo(df: pd.DataFrame, col: str = 'Promo') -> pd.DataFrame:
    """
    Count consecutive days a store has been in a promotion streak, including the current day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'Store', and `col` columns.
    col : str
        Name of the binary promo column to streak-count (e.g. 'Promo' or 'Promo2').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', 'Store', f'consecutive_{col.lower()}_days'].
    """
    result_dfs = []
    out_col = f'consecutive_{col.lower()}_days'

    for store_id, group in df[['Date', 'Store', col]].groupby('Store'):
        group = group.sort_values('Date')
        promo = group[col].values
        consecutive = np.zeros(len(promo), dtype=int)
        for i in range(len(promo)):
            if promo[i] == 1:
                consecutive[i] = (consecutive[i - 1] + 1) if i > 0 else 1

        result_dfs.append(pd.DataFrame({
            'Date': group['Date'].values,
            'Store': store_id,
            out_col: consecutive,
        }))

    return pd.concat(result_dfs, ignore_index=True)


def _target_encode(df: pd.DataFrame, cols: list, target: str) -> pd.DataFrame:
    """
    Time-aware target encoding: each (store, category value) pair is replaced by the
    expanding historical mean of `target` for that pair across all dates strictly before
    the current date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'Store', the columns in `cols`, and `target`.
    cols : list of str
        Categorical columns to encode.
    target : str
        Name of the target column to compute means from.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ['Date', 'Store'] with one `<col>_te` column per entry in `cols`.
    """
    result = df[['Date', 'Store']].copy()

    for col in cols:
        if col not in df.columns:
            continue
        # Use dict.fromkeys throughout to deduplicate keys when col == 'Store'
        group_keys  = list(dict.fromkeys(['Date', 'Store', col]))
        expand_keys = list(dict.fromkeys(['Store', col]))
        merge_on    = list(dict.fromkeys(['Date', 'Store', col]))

        # Aggregate to daily level per (date, store, category) to avoid within-day
        # duplicate rows biasing the expanding mean
        daily = (
            df.groupby(group_keys)[target]
            .mean()
            .reset_index()
            .sort_values('Date')
        )
        # shift(1) excludes the current date from its own encoding;
        # expanding mean is computed per (store, category value)
        daily[f'{col}_te'] = (
            daily.groupby(expand_keys)[target]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        col_te = (
            df[group_keys]
            .merge(daily[merge_on + [f'{col}_te']], on=merge_on, how='left')
            [['Date', 'Store', f'{col}_te']]
        )
        result = result.merge(col_te, on=['Date', 'Store'], how='left')

    return result.set_index(['Date', 'Store'])


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

    # Adjust lags, diffs, and roll_windows to account for the 1-day forward shift in
    # Shifted_Sales: Shifted_Sales[t] = Sales[t-1], so each offset already implies one
    # extra day of look-back. Subtracting 1 day from each offset's kwds restores the
    # original semantics (e.g. lag_days_7 still captures Sales 7 days before date t).
    def _adj(offset: DateOffset) -> DateOffset:
        kwds = dict(offset.kwds)
        kwds['days'] = kwds.get('days', 0) - 1
        return DateOffset(**kwds)

    lags = [_adj(lag) for lag in _to_list(lags)]
    diffs = [_adj(d) for d in _to_list(diffs)]
    roll_windows = {w: [_adj(lag) for lag in _to_list(window_lags)]
                    for w, window_lags in roll_windows.items()}

    sales_df = df[['Date', 'Store', 'Shifted_Sales']]

    # Competition-related features
    df['CompetitionDistance'] = df['CompetitionDistance'].apply(np.log1p)
    df['CompetitionSinceMonths'] = ( (df['Date'] - df['CompetitionSinceDate']).dt.days / 30.0 ).round()

    # Calendar and seasonality features
    df['is_weekend'] = df['Date'].dt.dayofweek >= 5
    df['DayOfMonth'] = df['Date'].dt.day
    df['is_month_start'] = (df['Date'].dt.day <= 3).astype(bool)
    df['is_month_end'] = (df['Date'].dt.day >= 28).astype(bool)

    # Basic date features
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter

    # Cyclical features
    cyclic_day_of_month = _make_cyclic(df['DayOfMonth'], period=31)
    cyclic_day_of_month.index = pd.MultiIndex.from_frame(df[['Date', 'Store']])
    cyclic_week_of_year = _make_cyclic(df['WeekOfYear'], period=52)
    cyclic_week_of_year.index = pd.MultiIndex.from_frame(df[['Date', 'Store']])
    
    # Lagged features
    lag_df = _make_lags(sales_df, lags).set_index(['Date', 'Store'])

    # Rolling window features
    window_dfs = []
    for window, lags in roll_windows.items():
        window_df = _make_rolling(sales_df, window, lags).set_index(['Date', 'Store'])
        window_dfs.append(window_df)

    # First-order differencing features
    diff_df = _make_differences(sales_df, diffs).set_index(['Date', 'Store'])

    # Holiday proximity features
    holiday_proximity_df = _make_holiday_proximity(df).set_index(['Date', 'Store'])

    # Consecutive promotion days
    consecutive_promo_df = _make_consecutive_promo(df, 'Promo').set_index(['Date', 'Store'])
    consecutive_promo2_df = _make_consecutive_promo(df, 'Promo2').set_index(['Date', 'Store'])

    # Merge everything (all feature DataFrames share the same (Date, Store) MultiIndex;
    # join() on the index avoids duplicate Date/Store columns that pd.merge would produce)
    feature_list = [cyclic_day_of_month, cyclic_week_of_year, lag_df, diff_df, holiday_proximity_df,
                    consecutive_promo_df, consecutive_promo2_df] + window_dfs
    merged = reduce(lambda left, right: left.join(right, how='outer'), feature_list).reset_index()
    df = df.merge(merged, on=['Date', 'Store'], how='left')

    # Target encoding of categorical features (time-aware: only historical dates used)
    cat_cols = ['Store', 'Promo', 'Promo2', 'SchoolHoliday',
                'Assortment', 'StoreType', 'StateHoliday',
                'DayOfWeek', 'Quarter', 'Year', 'Month']

    te_df = _target_encode(df, cat_cols, 'Shifted_Sales')
    df.drop([c for c in cat_cols if c != 'Store'], axis=1, inplace=True)
    df = df.merge(te_df.reset_index(), on=['Date', 'Store'], how='left')

    # Cleanup
    df.drop(['Promo2SinceDate', 'CompetitionSinceDate', 'DayOfMonth', 'WeekOfYear', 'Shifted_Sales'], axis=1, inplace=True)
    df.set_index(['Date', 'Store'], inplace=True)

    return df
