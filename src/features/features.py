""" 
This module contains the main function for computing features for the sales stores dataset. 
It combines various types of features, including past features (lags, differences, rolling means) 
and future features (calendar, competition, holidays, promotions) aligned with the forecast horizon. 
The compute function takes in the raw DataFrame and parameters for feature generation and returns a 
DataFrame with the generated features ready for modeling.
"""

import pandas as pd

from typing import List

from .calendar import calendar
from .timeseries import lag_features, diff_features, rolling_features
from .store import holiday_counters, days_with_competition, days_in_promotion

def compute(
        df: pd.DataFrame,
        lags: List[pd.DateOffset],
        diffs: List[pd.DateOffset],
        windows: List[str],
        horizon: pd.DateOffset) -> pd.DataFrame:
    """ 
    Generate features for the sales stores dataset. This includes both past features (lags, differences, rolling means)
    and future features (calendar, competition, holidays, promotions) aligned with the forecast horizon.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing sales and store information, indexed by Date 
        lags (List[pd.DateOffset]): A list of lag periods to use for generating lag features.
        diffs (List[pd.DateOffset]): A list of difference periods to use for generating difference features.
        windows (List[str]): A list of window sizes to use for generating rolling mean features.
        horizon (pd.DateOffset): The forecast horizon as a DateOffset.
    Returns:
        pd.DataFrame: A DataFrame containing the generated features, indexed by and Date.
    """
    
    features = [
        # Past features
        lag_features(df['Sales'], lags=lags),
        diff_features(df['Sales'], diffs=diffs),
        rolling_features(df['Sales'], windows=windows, agg_func='mean'),
        # Future features
        calendar(df.index.to_series() + horizon),
        days_with_competition(df['CompetitionStartDate'], offset=horizon),
        holiday_counters(df['isStateHoliday'], offset=horizon),
        days_in_promotion(df['Promo'], offset=horizon),
        days_in_promotion(df['Promo2'], offset=horizon)
    ]

    features_df = pd.concat(features, axis=1)
    
    return features_df
