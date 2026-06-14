
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


def holiday_counters(group: pd.Series, offset: pd.DateOffset) -> pd.DataFrame:
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


def competition_since_days(competition_since_date: pd.Series, offset: pd.DateOffset) -> pd.DataFrame:
    """Calculate the number of days since the competition started for each date.

    Args:
        competition_since_date (pd.Series): A pandas Series with datetime64[ns] values indicating the start date of the competition.
        offset (pd.DateOffset): A pandas DateOffset object representing the offset to apply to the dates.

    Returns:
        pd.DataFrame: A DataFrame with the number of days since the competition started, with NaN for dates before the competition started.
    """
    s = competition_since_date.copy()
    dates = s.index.to_series() + offset
    res = (dates - s).dt.days
    res = res.where(res >= 0, np.nan)
    res.name = 'CompetitionDaysSinceStart'
    return res.to_frame()  # Return as DataFrame to maintain consistency with other feature functions
