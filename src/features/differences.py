""" First-order difference features (absolute and relative change over a lag period). """

import pandas as pd

from typing import Iterable
from pandas.tseries.offsets import DateOffset

from .utils import _to_list, _pivot, _melt, _align


def _make_differences(df: pd.DataFrame,
                      diffs: int | Iterable[int]) -> pd.DataFrame:
    """
    Compute first-order differences of a time series for one or more lag periods,
    and return them in a long-format DataFrame aligned with the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in long format with three columns:
        [date_column, category_column, value_column].
    diffs : int or Iterable[int]
        One or more integer values indicating the number of days over which
        to compute first-order differences (e.g., `[1, 7, 30]` for daily,
        weekly, and monthly changes).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original identifying columns plus new
        difference feature columns named as:
        - `"lag_1_<n>_<unit>_diff"` for each value of `d` in `diffs`.
        - `"lag_1_<n>_<unit>_pct_change"` for each value of `d` in `diffs`.
    """

    diffs = _to_list(diffs)
    df_p = _pivot(df)
    feature_dfs = []
    for d in diffs:
        offset = d if isinstance(d, DateOffset) else DateOffset(days=d)
        d_name = "_".join(f"{v}_{k}" for k, v in offset.kwds.items())
        prior_index = df_p.index - offset
        df_p_prior = df_p.reindex(prior_index)
        df_p_prior.index = df_p.index
        feature_dfs.append(_melt(df_p - df_p_prior,                   f"lag_1_{d_name}_diff"))
        feature_dfs.append(_melt((df_p - df_p_prior).div(df_p_prior), f"lag_1_{d_name}_pct_change"))

    return _align(df, feature_dfs)
