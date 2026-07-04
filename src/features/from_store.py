"""
This module contains functions to create features related to store holidays and promotions for the Rossmann Store Sales
dataset. It includes functions to calculate the number of days to the next holiday, days since the last holiday,
days since competition started, and the number of consecutive promotion days. These features are essential for
modeling the sales patterns influenced by holidays and promotions.
"""

import pandas as pd
import numpy as np


def days_with_competition(competition_since_date: pd.Series, offset: pd.DateOffset) -> pd.Series:
    """Calculate the number of days since the competition started for each date.

    Args:
        competition_since_date (pd.Series): A pandas Series with datetime64[ns] values indicating the start date of the competition.
        offset (pd.DateOffset): A pandas DateOffset object representing the offset to apply to the dates.

    Returns:
        pd.Series: A pandas Series with the number of days since the competition started, with NaN for dates before the competition started.
    """
    s = competition_since_date.sort_index()

    # Shift the lookup dates by the specified offset to align with the forecast horizon, i.e.
    # if the forecast horizon is 7 days, we want to know how many days since competition started 
    # in 7 days from now - at the forecast date.
    lookup_dates = s.index.to_series() + offset
    res = (lookup_dates - s).dt.days
    res = res.where(res >= 0, np.nan)
    res.name = 'competition_days_since_start'

    res = res.reindex(competition_since_date.index)  # Ensure the result is aligned with the original index
    return res
    

def days_in_promotion(s: pd.Series, offset: pd.DateOffset) -> pd.Series:
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

    # Offset the lookup dates by the specified offset to align with the forecast horizon
    # i.e. if the forecast horizon is 7 days, we want to know how many consecutive promotions 
    # there will be in 7 days from now - at the forecast date.
    lookup_dates = s_sorted.index + offset
    result = consecutive_block.reindex(lookup_dates)
    result.index = s_sorted.index

    # Reorder the result to match the original order of the input series
    result = result.reindex(s.index)
    return result
