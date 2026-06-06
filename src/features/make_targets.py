import pandas as pd
from .lags import make_lags

def make_targets(df: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Generate multiple "future-shifted" target columns for each step in the forecast horizon.

    Args:
        df: A pandas Series with a MultiIndex of (Date, Store) and the target variable as values.
        horizon: The number of future time steps to forecast.

    Returns:
        A DataFrame with the same MultiIndex and new columns for each future target, named as '-1', '-2', ..., '-horizon'.
    """

    lags = range(-1, -(horizon + 1), -1)
    lag_df = make_lags(df.reset_index() , lags, names=[f'{-l}' for l in lags])
    lag_df.set_index(['Date', 'Store'], inplace=True)
    return lag_df
