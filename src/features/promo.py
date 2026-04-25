""" Promotion-related features: Promo2 flag, store data attachment, consecutive streak counts. """

import pandas as pd
import numpy as np


def in_promo2(row, date_col: str, interval_col: str, start_promo_date_col: str):
    """
    Determine whether a given observation falls within an active Promo2 period.

    Parameters
    ----------
    row : pd.Series
        A single row from a DataFrame, typically passed by `DataFrame.apply(axis=1)`.
    date_col : str
        The column name in `row` containing the date to check.
    interval_col : str
        The column name containing the active Promo2 intervals as strings,
        typically month abbreviations (e.g., "Feb,May,Aug,Nov").
    start_promo_date_col : str
        The column name containing the start date of the Promo2 campaign for the store.

    Returns
    -------
    bool
        True if the store is in an active Promo2 period for the given date, False otherwise.
    """

    month = row[date_col].strftime("%b")

    if pd.isna(month) or pd.isna(row[interval_col]):
        out = False
    else:
        out = (row[date_col] >= row[start_promo_date_col]) & (month in row[interval_col])

    return out


def attach_store_data(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge store-level metadata into the main DataFrame and compute active Promo2 flags.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame containing daily or transactional data.
        Must include at least the following columns:
        - `'Store'`: store identifier,
        - `'Date'`: observation date.
    stores : pd.DataFrame
        Store metadata DataFrame containing additional store-level attributes.
        Must include at least:
        - `'Store'`: store identifier (to join on),
        - `'PromoInterval'`: string listing active promo months (e.g., "Feb,May,Aug,Nov"),
        - `'Promo2SinceDate'`: datetime marking when Promo2 started for the store.

    Returns
    -------
    pd.DataFrame
        The input `df` enriched with store-level attributes and a new column:
        - `'Promo2'`: integer flag (1 if active Promo2, 0 otherwise).
    """

    df = df.merge(stores, on='Store')
    df['Promo2'] = df.apply(in_promo2, args=('Date', 'PromoInterval', 'Promo2SinceDate'), axis=1).astype(int)
    df.drop('PromoInterval', axis=1, inplace=True)

    return df


def make_consecutive_promo(df: pd.DataFrame, col: str = 'Promo') -> pd.DataFrame:
    """
    Count consecutive days a store has been in a promotion streak, including the current day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'Store', and `col` columns.
    col : str
        Name of the binary promo column to streak-count (e.g. 'Promo' or 'Promo2').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', 'Store', f'consecutive_{col.lower()}_days'].
    """
    result_dfs = []
    out_col = f'consecutive_{col.lower()}_days'

    for store_id, group in df[['Date', 'Store', col]].groupby('Store'):
        group = group.sort_values('Date')
        promo = group[col].values
        consecutive = np.zeros(len(promo), dtype=int)
        for i in range(len(promo)):
            if promo[i] == 1:
                consecutive[i] = (consecutive[i - 1] + 1) if i > 0 else 1

        result_dfs.append(pd.DataFrame({
            'Date': group['Date'].values,
            'Store': store_id,
            out_col: consecutive,
        }))

    return pd.concat(result_dfs, ignore_index=True)
