import pandas as pd

def in_promo2(row, date_col: str, interval_col: str, start_promo_date_col: str):
    """
    Determine whether a given store is in an active Promo2 period for a specific date.

    This function checks, for a given row in a DataFrame, whether the date in `date_col`
    falls within the active Promo2 intervals defined in `interval_col`, and occurs on or after
    the store's `start_promo_date_col`. It safely handles missing values.
    """

    month = row[date_col].strftime("%b")

    if pd.isna(month) or pd.isna(row[interval_col]):
        out = False
    else:
        out = (row[date_col] >= row[start_promo_date_col]) & (month in row[interval_col])

    return out

def attach_store_data(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:

    df = df.merge(stores, on='Store')
    df['Promo2'] = df.apply(in_promo2, args=('Date', 'PromoInterval', 'Promo2SinceDate'), axis=1).astype(int)
    df.drop('PromoInterval', axis=1, inplace=True)

    return df