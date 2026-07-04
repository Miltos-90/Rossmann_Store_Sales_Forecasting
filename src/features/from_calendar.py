""" 
This module contains functions to generate calendar features for time series forecasting. The main function, `date_features`, takes a series of dates and a forecast offset, and returns a DataFrame with various calendar features for the target forecast dates. These features include the year, quarter, month, week of the month, day of the week, and indicators for whether the date is the start or end of the month, or if it falls on a weekend or weekday.
"""

import pandas as pd

def calendar(dates: pd.Series) -> pd.DataFrame:
    """ 
    Generate calendar features for the target forecast dates, which are obtained by adding the forecast_offset to the input dates.

    Args:
        dates (pd.Series): A series of dates for which to generate features.

    Returns:
        pd.DataFrame: A DataFrame containing the generated calendar features for the target forecast dates.
    """

    features = {
        "year": dates.dt.year,
        "quarter": dates.dt.quarter,
        "month": dates.dt.month,
        "week_of_month": dates.dt.day // 7 + 1,
        "day_of_week": dates.dt.dayofweek,
        "is_month_start": dates.dt.is_month_start,
        "is_weekend": dates.dt.dayofweek >= 5,
        "is_weekday": dates.dt.dayofweek < 5,
        "is_month_end": dates.dt.is_month_end,
    }

    return pd.DataFrame.from_dict(features)
