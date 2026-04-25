""" Time-aware target encoding of categorical features, computed per (store, category) pair. """

import pandas as pd


def _target_encode(df: pd.DataFrame, cols: list, target: str) -> pd.DataFrame:
    """
    Time-aware target encoding: each (store, category value) pair is replaced by the
    expanding historical mean of `target` for that pair across all dates strictly before
    the current date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'Store', the columns in `cols`, and `target`.
    cols : list of str
        Categorical columns to encode.
    target : str
        Name of the target column to compute means from.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ['Date', 'Store'] with one `<col>_te` column per entry in `cols`.
    """
    result = df[['Date', 'Store']].copy()

    for col in cols:
        if col not in df.columns:
            continue
        # Use dict.fromkeys throughout to deduplicate keys when col == 'Store'
        group_keys  = list(dict.fromkeys(['Date', 'Store', col]))
        expand_keys = list(dict.fromkeys(['Store', col]))
        merge_on    = list(dict.fromkeys(['Date', 'Store', col]))

        # Aggregate to daily level per (date, store, category) to avoid within-day
        # duplicate rows biasing the expanding mean
        daily = (
            df.groupby(group_keys)[target]
            .mean()
            .reset_index()
            .sort_values('Date')
        )
        # shift(1) excludes the current date from its own encoding;
        # expanding mean is computed per (store, category value)
        daily[f'{col}_te'] = (
            daily.groupby(expand_keys)[target]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        col_te = (
            df[group_keys]
            .merge(daily[merge_on + [f'{col}_te']], on=merge_on, how='left')
            [['Date', 'Store', f'{col}_te']]
        )
        result = result.merge(col_te, on=['Date', 'Store'], how='left')

    return result.set_index(['Date', 'Store'])
