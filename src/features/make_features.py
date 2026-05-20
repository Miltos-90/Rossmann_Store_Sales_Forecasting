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

    sales_df = df[['Date', 'Store', 'Sales']]

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
    cyclic_day_of_month = make_cyclic(df['DayOfMonth'], period=31)
    cyclic_day_of_month.index = pd.MultiIndex.from_frame(df[['Date', 'Store']])
    cyclic_week_of_year = make_cyclic(df['WeekOfYear'], period=52)
    cyclic_week_of_year.index = pd.MultiIndex.from_frame(df[['Date', 'Store']])

    # Lagged features
    lag_df = make_lags(sales_df, lags).set_index(['Date', 'Store'])

    # Rolling window features
    window_dfs = []
    for window, lags in roll_windows.items():
        window_df = make_rolling(sales_df, window, lags).set_index(['Date', 'Store'])
        window_dfs.append(window_df)

    # First-order differencing features
    diff_df = make_differences(sales_df, diffs).set_index(['Date', 'Store'])

    # Holiday proximity features
    holiday_proximity_df = make_holiday_proximity(df).set_index(['Date', 'Store'])

    # Consecutive promotion days
    consecutive_promo_df = make_consecutive_promo(df, 'Promo').set_index(['Date', 'Store'])
    consecutive_promo2_df = make_consecutive_promo(df, 'Promo2').set_index(['Date', 'Store'])

    # Merge everything (all feature DataFrames share the same (Date, Store) MultiIndex;
    # join() on the index avoids duplicate Date/Store columns that pd.merge would produce)
    feature_list = [cyclic_day_of_month, cyclic_week_of_year, lag_df, diff_df, holiday_proximity_df,
                    consecutive_promo_df, consecutive_promo2_df] + window_dfs
    merged = reduce(lambda left, right: left.join(right, how='outer'), feature_list).reset_index()
    df = df.merge(merged, on=['Date', 'Store'], how='left')

    # Encode categorical features as pandas Categorical dtype
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.Categorical(df[col])

    # Cleanup: drop intermediate columns that are not needed for modeling, but keep 'Store' as a categorical feature.
    df.drop(['Promo2SinceDate', 'CompetitionSinceDate', 'DayOfMonth', 'WeekOfYear', 'Sales'], axis=1, inplace=True)
    df['Store_id'] = df['Store'].astype('category')
    df.set_index(['Date', 'Store'], inplace=True)

    return df
