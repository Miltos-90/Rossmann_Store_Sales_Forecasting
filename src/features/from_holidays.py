""" 
This module provides functions to retrieve holiday information for specific subdivisions 
of a country, as well as to determine the most likely subdivision for a store based on observed holiday data.
"""

import pandas as pd
import numpy as np

from holidays import country_holidays

EPS = np.sqrt(np.abs(np.finfo(float).eps))  # small number

def _subdivisions(country: str, years: list[int]) -> list[str]:
    """
    Get the list of subdivisions for a given country and years.

    Args:
        country (str): The country code (e.g., "DE" for Germany).
        years (list[int]): The list of years for which to retrieve subdivisions.

    Returns:
        list[str]: A list of subdivision codes for the specified country and years.
    """
    return list(country_holidays(country=country, years=years).subdivisions)

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


def _get_subdivision_holidays(
        country: str, language: str, subdiv: str, years: list[int], index_name: str
    ) -> pd.DataFrame:
    """ 
    Get holidays for a specific subdivision of a country for the given years.

    Args:
        country (str): The country code (e.g., "DE" for Germany).
        language (str): The language for holiday names.
        subdiv (str): The subdivision code (e.g., state or region code).
        years (list[int]): The list of years for which to retrieve holidays.
        index_name (str): The name to use for the index column in the returned DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing holidays for the specified subdivision and years.
                      Columns include 'Date', 'name', and 'subdiv'.
    """
    
    # Dictionary of holidays for the given country, subdivision, and years.
    # keys are dates (datetime.date), values are holiday names (str)
    subdiv_holiday_dict = country_holidays(country = country, 
                                           language=language,
                                           subdiv=subdiv,
                                           years=years)
    
    subdiv_holiday_df = pd.DataFrame.from_dict(subdiv_holiday_dict,
                                            orient='index', 
                                            columns=['name'])
    
    subdiv_holiday_df["subdiv"] = subdiv  # Add a column for the subdivision code

    subdiv_holiday_df = (subdiv_holiday_df.reset_index()
                         .rename(columns={'index': index_name}))

    return subdiv_holiday_df


def _get_country_holidays(
        years: list[int],
        country: str,
        language: str,
        index_name: str) -> pd.DataFrame:
    """
    Get holidays for all subdivisions of a country for the given years.

    Args:
        years (list[int]): The list of years for which to retrieve holidays.
        country (str): The country code (e.g., "DE" for Germany).
        language (str): The language for holiday names.
        index_name (str): The name to use for the index column in the returned DataFrame.
        
    Returns:
        pd.DataFrame: A DataFrame containing holidays for all subdivisions.
                      Columns include 'Date', 'name', and 'subdiv'.
    """
    
    # Get all subdivision codes for the specified country
    subdivision_codes = _subdivisions(country, years)

    subdiv_holiday_df = []
    for subdiv in subdivision_codes:
        subdiv_holidays = _get_subdivision_holidays(country, language, subdiv, years, index_name)
        subdiv_holiday_df.append(subdiv_holidays)

    subdiv_holiday_df = pd.concat(subdiv_holiday_df)
    subdiv_holiday_df['Date'] = pd.to_datetime(subdiv_holiday_df['Date'])

    return subdiv_holiday_df


def subdivision(holiday_df: pd.Series, country: str, language: str, index_name: str) -> str:
    """
    Determine the most likely subdivision for a given store based on holiday agreement.

    Args:
        holiday_df (pd.DataFrame): DataFrame containing 'Date', and 'isStateHoliday' columns for the store's observed holiday data.
        country (str): The country code (e.g., "DE" for Germany).
        language (str): The language for holiday names.
        index_name (str): The name to use for the index column in the returned DataFrame.

    Returns:
        str: The subdivision code that has the highest agreement with the observed holiday data.
    """
    
    # Get the unique years present in the dataset for the specific store
    years_in_dset = holiday_df['Date'].dt.year.unique().tolist()

    # Get all subdivision codes for the specified country
    subdiv_codes = _subdivisions(country, years_in_dset)

    # Create a holiday indicator DataFrame where each row corresponds to a date and 
    # each column corresponds to a subdivision. The values are 1 if the date is a 
    # holiday in that subdivision, and 0 otherwise.
    holiday_indicator = (
        _get_country_holidays(years_in_dset, country, language, index_name)
        .pivot(index='Date', columns='subdiv', values='name')
        .notna()
        .reset_index()
    )

    # For each row, attach the holiday indicator for the corresponding date. 
    # If a date is a holiday in any subdivision, it will be marked as True; otherwise, it will be False.
    merged = holiday_df.merge(holiday_indicator, on='Date', how='left')
    merged[subdiv_codes] = merged[subdiv_codes].fillna(0)  # Fill NaN values with 0 and convert to int8

    # In the following, N is the number of observations/rows in the dataset, 
    # and M is the number of subdivisions (columns) in the holiday indicator.
    observed   = merged["isStateHoliday"].to_numpy() # (N,)   observed
    candidates = merged[subdiv_codes]  # (N,M) candidates

    # Compute the per-row agreement between the observed and candidate holiday indicators
    # aggrement = True if the observed value matches the candidate value for that row and subdivision, False otherwise
    agreement = (candidates.values == observed[:, None])  # (N, M) agreement

    # For each store and subdivision, compute the overall agreement across all rows (dates)
    scores = (pd.DataFrame(agreement, columns=subdiv_codes).sum())  # (M) scores
    most_likely_subdivision = scores.idxmax()  # Get the subdivision code corresponding to the maximum score

    return most_likely_subdivision


def holiday_names(
        dates: pd.Series, 
        country: str,
        subdivision: str,
        language: str,
        offset: pd.DateOffset
        ) -> pd.DataFrame:
    """
    Get holidays for a specific store based on its subdivision and the given dates.

    Args:
        dates (pd.Series): Series of dates for which to retrieve holidays.
        country (str): The country code (e.g., "DE" for Germany).
        language (str): The language for holiday names.
        subdivision (str): The subdivision code for the store.
        offset (pd.DateOffset): A pandas DateOffset object representing the forecast horizon.
                                The lookup is performed at dates + offset, but the result is
                                indexed by the original dates.

    Returns:
        pd.DataFrame: A DataFrame containing the holiday names for each date in the list.
                      Columns include the index_name and 'holiday_name'.
    """

    # Shift the lookup dates by the forecast horizon to determine which holiday falls
    # at dates + offset, while keeping the original dates as the output index.
    lookup_dates = dates + offset

    # Extract unique years from the lookup dates
    years_in_dset = lookup_dates.dt.year.unique().tolist()

    # Get holidays for the specific subdivision and years
    holiday_df = _get_subdivision_holidays(country=country,
                                          language=language,
                                          subdiv=subdivision,
                                          years=years_in_dset,
                                          index_name=dates.name)

    # Dataframe with a single column "holiday_name" that contains 
    # the holiday name for each date in the group.
    holiday_df = (holiday_df
                  .set_index(dates.name)
                  ['name']
                  .reindex(lookup_dates)
                  .set_axis(dates)
                  .rename("holiday_name")
                  .to_frame())
    
    holiday_df.index.name = dates.name  # Set the index name for clarity

    return holiday_df


def holiday_waves(group: pd.Series, offset: pd.DateOffset, sigma: float) -> pd.DataFrame:
    """ 
    Calculate holiday-related features for a given group of dates.

    Args:
        group (pd.Series): A pandas Series with boolean values indicating holidays.
        offset (pd.DateOffset): A pandas DateOffset object representing the forecast horizon.
        sigma (float): Standard deviation for the Gaussian function used in the holiday wave calculation.
    
    Returns:
        pd.DataFrame: A DataFrame with columns 'pre_holiday_wave' and 'post_holiday_wave'.
    """

    group_sorted = group.sort_index()
    is_holiday   = group_sorted == True
    holiday_days = _dates_to_days(group_sorted.index[is_holiday])

    # Shift the lookup dates by the forecast horizon to compute days to next and since last holiday
    # from those dates. This is important because we want to know how many days until the next holiday 
    # and how many days since the last holiday, relative to the forecast horizon, not the current date.
    lookup_days = _dates_to_days(group_sorted.index + offset)  # shift only the lookup dates

    if len(holiday_days) > 0:
        days_to_next    = _days_to_holiday(holiday_days, lookup_days)
        days_since_last = _days_since_holiday(holiday_days, lookup_days)
    else:
        # Arrays filled with NaN values
        days_to_next    = np.full(len(group), np.nan)
        days_since_last = np.full(len(group), np.nan)

    # Compute the Gaussian holiday wave effect based on the days to next and since last holiday.
    denom = 2 * sigma ** 2  # Denominator for the Gaussian function
    post_holiday_wave = np.exp(- days_since_last ** 2 / denom)
    pre_holiday_wave  = np.exp(- days_to_next ** 2 / denom)

    waves = np.stack([pre_holiday_wave, post_holiday_wave], axis=-1)
    waves[np.abs(waves) < EPS] = 0

    res = pd.DataFrame(data=waves,
                       index=group_sorted.index,
                       columns=['pre_holiday_wave', 'post_holiday_wave'])

    return res.reindex(group.index)