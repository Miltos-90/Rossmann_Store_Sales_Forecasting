""" Main feature-engineering entry point: make_features. """

import pandas as pd
import numpy as np

from typing import Iterable, Dict
from functools import reduce

from .utils import to_list
from .lags import make_lags
from .rolling import make_rolling
from .differences import make_differences
from .cyclic import make_cyclic
from .holidays import make_holiday_proximity
from .promo import make_consecutive_promo

# List of categorical features to convert to pandas Categorical dtype after feature generation.
# Some derived features might not exist in the output Dataframe, 
# so we check for their presence before conversion.
CATEGORICAL_FEATURES = ['Promo', 'Promo2', 'SchoolHoliday',
                        'Assortment', 'StoreType', 'StateHoliday',
                        'DayOfWeek', 'Quarter', 'Year', 'Month', 'Open',
                        'is_month_end', 'is_month_start', 'is_weekend']

DAYS_IN_MONTH = 30
WEEKS_IN_YEAR = 52

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
        The index should be a MultiIndex of (Date, Store) or the DataFrame should have 'Date' and 'Store' columns.
    lags : int or Iterable[int]
        List (or single value) of lag periods (in days) to compute past sales values.
        For example, `[1, 7, 30]` creates features for sales 1, 7, and 30 days ago.
    roll_windows : Dict of int and int or Iterable[int]
        Dictionary of rolling window sizes (in days) and corresponding lags (in days) to compute
        moving averages and other rolling statistics of past sales.
    diffs : int or Iterable[int]
        List (or single value) of differencing periods (in days) for computing
        first-order change features (e.g., daily or weekly sales deltas).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional engineered features, including:
    """
    lags = to_list(lags)
    diffs = to_list(diffs)
    roll_windows = {w: to_list(window_lags) for w, window_lags in roll_windows.items()}

    df_ = df.reset_index()  # Ensure 'Date' and 'Store' are columns for merging later
    sales_df = df_[['Date', 'Store', 'Sales']]

    """
    # Competition-related features
    df_['CompetitionDistance'] = df_['CompetitionDistance'].apply(np.log1p)
    df_['CompetitionSinceMonths'] = ( (df_['Date'] - df_['CompetitionSinceDate']).dt.days / 30.0 ).round()

    # Calendar and seasonality features
    df_['is_weekend'] = df_['Date'].dt.dayofweek >= 5
    df_['DayOfMonth'] = df_['Date'].dt.day
    df_['is_month_start'] = (df_['Date'].dt.day <= 3).astype(bool)
    df_['is_month_end'] = (df_['Date'].dt.day >= 28).astype(bool)

    # Basic date features
    df_['WeekOfYear'] = df_['Date'].dt.isocalendar().week
    df_['Month'] = df_['Date'].dt.month
    df_['Year'] = df_['Date'].dt.year
    df_['Quarter'] = df_['Date'].dt.quarter

    # Cyclical features
    cyclic_day_of_month = make_cyclic(df_['DayOfMonth'], period=DAYS_IN_MONTH)
    cyclic_day_of_month.index = pd.MultiIndex.from_frame(df_[['Date', 'Store']])
    cyclic_week_of_year = make_cyclic(df_['WeekOfYear'], period=WEEKS_IN_YEAR)
    cyclic_week_of_year.index = pd.MultiIndex.from_frame(df_[['Date', 'Store']])

    # Lagged features
    if lags:
        lag_df = make_lags(sales_df, lags).set_index(['Date', 'Store'])
    else:
        lag_df = pd.DataFrame(index=pd.MultiIndex.from_frame(df_[['Date', 'Store']]))  # Empty DataFrame if no lags specified


    if roll_windows:
        window_dfs = []
        for window, window_lags in roll_windows.items():
            window_df = make_rolling(sales_df, window, window_lags).set_index(['Date', 'Store'])
            window_dfs.append(window_df)
    else:
        window_dfs = []  # No rolling features if roll_windows is empty

    # First-order differencing features
    if diffs:
        diff_df = make_differences(sales_df, diffs).set_index(['Date', 'Store'])
    else:
        diff_df = pd.DataFrame(index=pd.MultiIndex.from_frame(df_[['Date', 'Store']]))  # Empty DataFrame if no diffs specified

    # Holiday proximity features
    holiday_proximity_df = make_holiday_proximity(df_).set_index(['Date', 'Store'])
    """

    # Consecutive promotion days
    consecutive_promo_df = make_consecutive_promo(df_, 'Promo').set_index(['Date', 'Store'])
    consecutive_promo2_df = make_consecutive_promo(df_, 'Promo2').set_index(['Date', 'Store'])

    # Merge everything (all feature DataFrames share the same (Date, Store) MultiIndex;
    # join() on the index avoids duplicate Date/Store columns that pd.merge would produce)
    feature_list = [cyclic_day_of_month, cyclic_week_of_year, lag_df, diff_df, holiday_proximity_df,
                    consecutive_promo_df, consecutive_promo2_df] + window_dfs
    merged = reduce(lambda left, right: left.join(right, how='outer'), feature_list).reset_index()
    df_ = df_.merge(merged, on=['Date', 'Store'], how='left')

    # Encode categorical features as pandas Categorical dtype
    for col in CATEGORICAL_FEATURES:
        if col in df_.columns:
            df_[col] = pd.Categorical(df_[col])

    # Cleanup: drop intermediate columns that are not needed for modeling, but keep 'Store' as a categorical feature.
    cleanup_cols = ['Promo2SinceDate', 'CompetitionSinceDate', 'Open', 'DayOfMonth', 'WeekOfYear', 'Sales']
    df_.drop(cleanup_cols, axis=1, inplace=True)
    df_['Store_id'] = df_['Store'].astype('category')
    df_.set_index(['Date', 'Store'], inplace=True)

    return df_
