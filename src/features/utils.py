""" Shared low-level helpers used across feature sub-modules. """

import pandas as pd

from typing import Iterable
from pandas.tseries.offsets import DateOffset


def to_list(x):
    """
    Normalize input to a list.

    Parameters
    ----------
    x : int or Iterable
        Input value or iterable.

    Returns
    -------
    list
        List containing the input(s).
    """
    return [x] if isinstance(x, (int, DateOffset)) else list(x)


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-format DataFrame to wide format, sorted by index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least three columns.

    Returns
    -------
    pd.DataFrame
        Pivoted and sorted DataFrame.
    """
    return (pd.pivot(df, index=df.columns[0], columns=df.columns[1], values=df.columns[2])
            .sort_index())


def melt(df_wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Melt a wide-format DataFrame back to long format with a specific value column name.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format DataFrame (pivoted).
    name : str
        Name for the value column.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with MultiIndex.
    """
    index_name = df_wide.index.name
    columns_name = df_wide.columns.name
    return (df_wide.reset_index()
            .melt(id_vars=[index_name], value_name=name)
            .set_index([index_name, columns_name]))


def align(df: pd.DataFrame, feature_dfs: list) -> pd.DataFrame:
    """
    Merge feature DataFrames with the original DataFrame, preserving row order and columns.

    Parameters
    ----------
    df : pd.DataFrame
        Original long-format DataFrame.
    feature_dfs : list of pd.DataFrame
        List of feature DataFrames to merge.

    Returns
    -------
    pd.DataFrame
        DataFrame with features aligned to the original rows.
    """
    feature_df = pd.concat(feature_dfs, axis=1).reset_index()
    return df.merge(feature_df, how='left').loc[:, feature_df.columns]
