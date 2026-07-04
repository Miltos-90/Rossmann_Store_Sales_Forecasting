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
from src.features.from_calendar import calendar
from src.features.from_target import lag_features, diff_features, rolling_features
from src.features.from_store import days_with_competition, days_in_promotion
from src.features.from_holidays import holiday_waves, subdivision, holiday_names

def compute(df: pd.DataFrame, 
            config: FeatureEngineeringSettings, 
            horizon: pd.DateOffset) -> pd.DataFrame:
    """ 
    Generate features for the sales stores dataset. This includes both past features (lags, differences, rolling means)
    and future features (calendar, competition, holidays, promotions) aligned with the forecast horizon.

    Args:
        df (pd.DataFrame): The input DataFrame containing sales and store information.
                           - Indexed by 'Date' of type datetime64[ns].
                           - Columns include 'Sales', 'isStateHoliday', 'CompetitionStartDate', 'Promo', and 'Promo2'.
        config (FeatureEngineeringSettings): The configuration settings for feature engineering, including lags, diffs, windows, and horizon.
        horizon (pd.DateOffset): The forecast horizon for aligning future features.
    Returns:
        pd.DataFrame: A DataFrame containing the generated features, indexed by 'Date'.
    """

    log_sales = np.log1p(df['Sales'])  # Log-transform the sales to stabilize variance

    # This returns a string, so we will add it to the features dataframe at the end.
    subdiv = subdivision(df['isStateHoliday'].reset_index(),
                         country=config.holidays["country"],
                         language=config.holidays["language"],
                         index_name=df.index.name)  # Determine the subdivision based on holiday data

    features = [
        # Past features - these are aligned with the current date as they depend on past data
        lag_features(log_sales, lags=config.lags),
        diff_features(log_sales, diffs=config.diffs),
        rolling_features(log_sales, windows=config.windows, agg_func='mean'),

        # Future features - these are aligned with the forecast horizon date as they are known in advance
        calendar(df.index.to_series() + horizon),
        days_with_competition(df['CompetitionStartDate'], offset=horizon),
        holiday_waves(df['isStateHoliday'], offset=horizon, sigma=config.holidays["sigma"]),
        holiday_names(df.index.to_series(),
                      country=config.holidays["country"],
                      language=config.holidays["language"],
                      subdivision=subdiv,
                      offset=horizon),
        days_in_promotion(df['Promo'], offset=horizon),
        days_in_promotion(df['Promo2'], offset=horizon)
    ]

    features_df = pd.concat(features, axis=1)
    features_df["subdivision"] = subdiv # Now, add the subdivision for all rows.
    
    return features_df
