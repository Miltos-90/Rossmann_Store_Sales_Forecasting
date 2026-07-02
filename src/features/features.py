""" 
This module contains the main function for computing features for the sales stores dataset. 
It combines various types of features, including past features (lags, differences, rolling means) 
and future features (calendar, competition, holidays, promotions) aligned with the forecast horizon. 
The compute function takes in the raw DataFrame and parameters for feature generation and returns a 
DataFrame with the generated features ready for modeling.
"""

import pandas as pd
import numpy as np

from src.settings import FeatureEngineeringSettings
from src.features.calendar import calendar
from src.features.timeseries import lag_features, diff_features, rolling_features
from src.features.store import holiday_counters, days_with_competition, days_in_promotion

def compute(df: pd.DataFrame, 
            config: FeatureEngineeringSettings, 
            horizon: pd.DateOffset) -> pd.DataFrame:
    """ 
    Generate features for the sales stores dataset. This includes both past features (lags, differences, rolling means)
    and future features (calendar, competition, holidays, promotions) aligned with the forecast horizon.

    Args:
        df (pd.DataFrame): The input DataFrame containing sales and store information, indexed by Dateand a column named 'Sales'.
        config (FeatureEngineeringSettings): The configuration settings for feature engineering, including lags, diffs, windows, and horizon.
        horizon (pd.DateOffset): The forecast horizon for aligning future features.
    Returns:
        pd.DataFrame: A DataFrame containing the generated features, indexed by and Date.
    """

    log_sales = np.log1p(df['Sales'])  # Log-transform the sales to stabilize variance

    features = [
        # Past features - these are aligned with the current date as they depend on past data
        lag_features(log_sales, lags=config.lags),
        diff_features(log_sales, diffs=config.diffs),
        rolling_features(log_sales, windows=config.windows, agg_func='mean'),
        # Future features - these are aligned with the forecast horizon date as they are known in advance
        calendar(df.index.to_series() + horizon),
        days_with_competition(df['CompetitionStartDate'], offset=horizon),
        holiday_counters(df['isStateHoliday'], offset=horizon, sigma=config.holidays["sigma"]),
        days_in_promotion(df['Promo'], offset=horizon),
        days_in_promotion(df['Promo2'], offset=horizon)
    ]

    features_df = pd.concat(features, axis=1)
    
    return features_df
