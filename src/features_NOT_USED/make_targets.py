import pandas as pd
from .lags import make_lags

def make_targets(df: pd.Series, horizon: int) -> pd.Series:
    """
    Generate multiple "future-shifted" target columns for each step in the forecast horizon.

    Args:
        df: A pandas Series with a MultiIndex of (Date, Store) and the target variable as values.
        horizon: The number of future time steps to forecast.

    Returns:
        A Series with the same MultiIndex and the future target values for the specified horizon.
    """

    lag_df = make_lags(df.reset_index(), lags=[-horizon], names=[str(horizon)])
    lag_df = lag_df.set_index(['Date', 'Store'])[str(horizon)]

    return lag_df
