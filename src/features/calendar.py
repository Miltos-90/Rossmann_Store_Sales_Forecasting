""" 
This module contains functions to generate calendar features for time series forecasting. The main function, `date_features`, takes a series of dates and a forecast offset, and returns a DataFrame with various calendar features for the target forecast dates. These features include the year, quarter, month, week of the month, day of the week, and indicators for whether the date is the start or end of the month, or if it falls on a weekend or weekday.
"""

import pandas as pd

def calendar(dates: pd.Series, forecast_offset: pd.DateOffset) -> pd.DataFrame:
    """ 
    Generate calendar features for the target forecast dates, which are obtained by adding the forecast_offset to the input dates.

    Args:
        dates (pd.Series): A series of dates for which to generate features.
        forecast_offset (pd.DateOffset): The offset to apply to the input dates to get the target forecast dates.

    Returns:
        pd.DataFrame: A DataFrame containing the generated calendar features for the target forecast dates.
    """
    
    fdates = dates.apply(lambda x: x + forecast_offset)

    features = {
        "year": fdates.dt.year,
        "quarter": fdates.dt.quarter,
        "month": fdates.dt.month,
        "week_of_month": fdates.dt.day // 7 + 1,
        "day_of_week": fdates.dt.dayofweek,
        "is_month_start": fdates.dt.is_month_start,
        "is_weekend": fdates.dt.dayofweek >= 5,
        "is_weekday": fdates.dt.dayofweek < 5,
        "is_month_end": fdates.dt.is_month_end,
    }

    return pd.DataFrame.from_dict(features)
