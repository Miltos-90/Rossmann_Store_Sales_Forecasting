""" Holiday proximity features: days to/since state and school holidays. """

import pandas as pd
import numpy as np


def make_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proximity in days to the nearest state and school holidays per store.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'Store', 'StateHoliday', 'SchoolHoliday' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', 'Store', 'days_to_next_state_holiday',
        'days_since_last_state_holiday', 'days_to_next_school_holiday'].
    """
    ns_per_day = 86_400 * 10 ** 9  # nanoseconds in a day; used to convert int64 timestamp differences back to days
    result_dfs = []

    for store_id, group in df[['Date', 'Store', 'StateHoliday', 'SchoolHoliday']].groupby('Store'):
        group = group.sort_values('Date')
        # Convert dates to int64 nanoseconds so we can do fast arithmetic with numpy
        dates_ns = group['Date'].values.astype('int64')

        # StateHoliday is '0' (string) or 0 (int) when there is no holiday; anything else is a real holiday
        is_state_hol = ~group['StateHoliday'].isin(['0', 0]) & group['StateHoliday'].notna()
        state_hol_ns = group.loc[is_state_hol, 'Date'].values.astype('int64')  # sorted holiday timestamps

        is_school_hol = group['SchoolHoliday'] == 1
        school_hol_ns = group.loc[is_school_hol, 'Date'].values.astype('int64')  # sorted holiday timestamps

        if len(state_hol_ns) > 0:
            # searchsorted(side='left') gives the index of the first holiday >= current date
            idx_next = np.searchsorted(state_hol_ns, dates_ns, side='left')
            has_next = idx_next < len(state_hol_ns)  # False for dates after the last known holiday
            days_to_next_state = np.where(
                has_next,
                (state_hol_ns[np.minimum(idx_next, len(state_hol_ns) - 1)] - dates_ns) / ns_per_day,
                np.nan)  # NaN when no future holiday exists in the dataset

            # searchsorted(side='right') - 1 gives the index of the last holiday <= current date
            idx_prev = np.searchsorted(state_hol_ns, dates_ns, side='right') - 1
            has_prev = idx_prev >= 0  # False for dates before the first known holiday
            days_since_last_state = np.where(
                has_prev,
                (dates_ns - state_hol_ns[np.maximum(idx_prev, 0)]) / ns_per_day,
                np.nan)  # NaN when no past holiday exists in the dataset
        else:
            days_to_next_state = np.full(len(group), np.nan)
            days_since_last_state = np.full(len(group), np.nan)

        if len(school_hol_ns) > 0:
            idx_next = np.searchsorted(school_hol_ns, dates_ns, side='left')
            has_next = idx_next < len(school_hol_ns)
            days_to_next_school = np.where(
                has_next,
                (school_hol_ns[np.minimum(idx_next, len(school_hol_ns) - 1)] - dates_ns) / ns_per_day,
                np.nan)
        else:
            days_to_next_school = np.full(len(group), np.nan)

        result_dfs.append(pd.DataFrame({
            'Date': group['Date'].values,
            'Store': store_id,
            'days_to_next_state_holiday': days_to_next_state,
            'days_since_last_state_holiday': days_since_last_state,
            'days_to_next_school_holiday': days_to_next_school,
        }))

    return pd.concat(result_dfs, ignore_index=True)
