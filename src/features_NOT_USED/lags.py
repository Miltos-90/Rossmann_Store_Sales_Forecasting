""" Lag features and forecast target generation. """

import pandas as pd

from typing import Iterable
from pandas.tseries.offsets import DateOffset

from .utils import to_list, pivot, melt, align


def make_lags(df: pd.DataFrame, lags: int | Iterable[int], names: list[str] = None) -> pd.DataFrame:
    """
    Generate and merge multiple lagged feature DataFrames based on one or more DateOffset objects.

    Parameters
    ----------
    df : pd.DataFrame
        Input long-format DataFrame containing at least three columns:
        1. A time or index column (used for pivot index),
        2. A category or variable column (used for pivot columns),
        3. A value column (used for pivot values).
    lags : int or Iterable[int]
        One or more pandas DateOffset objects specifying the temporal lags to apply.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the original columns (apart from the value column)
        plus additional lag feature columns
        (e.g., `"lag_days_7"`, `"lag_weeks_2"`, etc.), merged by date and key.

    Notes
    -----
    - The input DataFrame must be sorted in ascending time order.
    """

    lags = to_list(lags)
    df_p = pivot(df)
    lag_dfs = []
    for lag in lags:
        offset = lag if isinstance(lag, DateOffset) else DateOffset(days=lag)
        prior_index = df_p.index - offset
        df_p_lagged = df_p.reindex(prior_index)
        df_p_lagged.index = df_p.index

        if names is not None:
            name = names.pop(0)
        else:
            name = "_".join(f"lag_{v}_{k}" for k, v in offset.kwds.items())

        lag_dfs.append(melt(df_p_lagged, name))

    return align(df, lag_dfs)
