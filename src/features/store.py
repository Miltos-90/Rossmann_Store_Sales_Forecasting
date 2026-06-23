"""
This module contains functions to create features related to store holidays and promotions for the Rossmann Store Sales
dataset. It includes functions to calculate the number of days to the next holiday, days since the last holiday,
days since competition started, and the number of consecutive promotion days. These features are essential for
modeling the sales patterns influenced by holidays and promotions.
"""

import pandas as pd
import numpy as np


def _dates_to_days(dates: pd.Series) -> np.ndarray:
    """
    Convert a series of dates to integer days since epoch.
    
    Args: 
        dates (pd.Series): A pandas Series of datetime64[ns] dates.

    Returns:
        np.ndarray: An array of int64 representing days since epoch.
    """
    return dates.to_numpy().astype('datetime64[D]').astype('int64')


def _days_to_holiday(holiday_days: np.ndarray, dates_days: np.ndarray) -> np.ndarray:
    """
    Calculate the number of days to the next holiday for each date.

    Args:
        holiday_days (np.ndarray): An array of int64 representing holiday days since epoch.
        dates_days (np.ndarray): An array of int64 representing dates since epoch.

    Returns:
        np.ndarray: An array of int64 representing days to the next holiday.
    """
    idx_next = np.searchsorted(holiday_days, dates_days, side='left')
    is_next_holiday = idx_next < len(holiday_days)
    days_to_next_holiday = np.where(
        is_next_holiday,
        holiday_days[idx_next.clip(max=len(holiday_days) - 1)] - dates_days,
        np.nan)
    return days_to_next_holiday


def _days_since_holiday(holiday_days: np.ndarray, dates_days: np.ndarray) -> np.ndarray:
    """
    Calculate the number of days since the last holiday for each date.

    Args:
        holiday_days (np.ndarray): An array of int64 representing holiday days since epoch.
        dates_days (np.ndarray): An array of int64 representing dates since epoch.

    Returns:
        np.ndarray: An array of int64 representing days since the last holiday.
    """
    idx_prev = np.searchsorted(holiday_days, dates_days, side='right') - 1
    is_prev_holiday = idx_prev >= 0
    days_since_last_holiday = np.where(
        is_prev_holiday,
        dates_days - holiday_days[idx_prev.clip(min=0)],
        np.nan)
    return days_since_last_holiday


def holiday_counters(group: pd.Series, offset: pd.DateOffset = pd.DateOffset(0)) -> pd.DataFrame:
    """ 
    Calculate holiday-related features for a given group of dates.

    Args:
        group (pd.Series): A pandas Series with boolean values indicating holidays.
        offset (pd.DateOffset): A pandas DateOffset object representing the forecast horizon.

    Returns:
        pd.DataFrame: A DataFrame with columns 'DaysToNextHoliday' and 'DaysSinceLastHoliday'.
    """
    group_sorted = group.sort_index()
    is_holiday = group_sorted == True
    holiday_days = _dates_to_days(group_sorted.index[is_holiday])
    lookup_days  = _dates_to_days(group_sorted.index + offset)  # shift only the lookup dates

    if len(holiday_days) > 0:
        days_to_next   = _days_to_holiday(holiday_days, lookup_days)
        days_since_last = _days_since_holiday(holiday_days, lookup_days)
    else:
        days_to_next = np.full(len(group), np.nan)  # Array filled with NaN values
        days_since_last = np.full(len(group), np.nan)

    res = pd.DataFrame(data=np.stack([days_to_next, days_since_last], axis=-1),
                       index=group_sorted.index,
                       columns=['DaysToNextHoliday', 'DaysSinceLastHoliday'])

    return res.reindex(group.index)


def days_with_competition(competition_since_date: pd.Series, offset: pd.DateOffset = pd.DateOffset(0)) -> pd.Series:
    """Calculate the number of days since the competition started for each date.

    Args:
        competition_since_date (pd.Series): A pandas Series with datetime64[ns] values indicating the start date of the competition.
        offset (pd.DateOffset): A pandas DateOffset object representing the offset to apply to the dates.

    Returns:
        pd.Series: A pandas Series with the number of days since the competition started, with NaN for dates before the competition started.
    """
    s = competition_since_date.sort_index()
    dates = s.index.to_series() + offset
    res = (dates - s).dt.days
    res = res.where(res >= 0, np.nan)
    res.name = 'CompetitionDaysSinceStart'

    res = res.reindex(competition_since_date.index)  # Ensure the result is aligned with the original index
    return res
    

def days_in_promotion(s: pd.Series, offset: pd.DateOffset = pd.DateOffset(0)) -> pd.Series:
    """ Calculate the number of consecutive promotions for each date in the series.
    
    Args:
        s (pd.Series): A pandas Series representing the promotion status (1 for active, 0 for inactive).
        offset (pd.DateOffset): A DateOffset to shift the lookup dates (e.g. forecast horizon).

    Returns:
        pd.Series: A pandas Series with the same index as the input, containing the number of consecutive promotions.
    """

    # Ensure the series is sorted by index from least recent to most recent
    s_sorted = s.sort_index()

    # Create a boolean series where the promotion is active
    is_promo_active = s_sorted == 1

    # Generate unique IDs for each block of consecutive promotions
    block_id = (~is_promo_active).cumsum()

    # Group by the blocks and calculate the cumulative sum
    consecutive_block = is_promo_active.groupby(block_id).cumsum()

    # Look up the consecutive count at (index + offset) and restore original index
    lookup_index = s_sorted.index + offset
    result = consecutive_block.reindex(lookup_index)
    result.index = s_sorted.index

    # Reorder the result to match the original order of the input series
    result = result.reindex(s.index)
    return result
