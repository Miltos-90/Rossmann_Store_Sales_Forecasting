import pandas as pd

def date_from_week_year(row, year_col: str, week_col: str, day: int = 1):
    """
    Convert (year, week) to a date, handling missing values.
    Returns Monday of that ISO week by default.
    """

    if pd.isna(row[year_col]) or pd.isna(row[week_col]):
        out = pd.NaT  # Missing date

    else:
        date_str = f"{int(row[year_col])}-W{int(row[week_col]):02d}-{day}"
        out = pd.to_datetime(date_str, format="%G-W%V-%u")

    return out

def date_from_month_year(row, year_col: str, month_col: str, day: int = 1):
    """
    Convert (year, month) to a date, handling missing values.
    Returns Monday of that ISO week by default.
    """

    if pd.isna(row[year_col]) or pd.isna(row[month_col]):
        out = pd.NaT  # Missing date

    else:
        date_str = f"{int(row[year_col])}-{int(month_col):02d}-{day:02d}"
        out = pd.to_datetime(date_str, format="%G-W%V-%u")

    return out

def store_data(df):

    args = ('Promo2SinceYear', 'Promo2SinceWeek')
    df['Promo2SinceDate'] = df.apply(date_from_week_year, args=args, axis=1)

    args = ('CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth')
    df['CompetitionSinceDate'] = df.apply(date_from_week_year, args=args, axis=1)

    remove_cols = ['Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'] # not needed anymore
    df.drop(remove_cols, axis=1, inplace=True)

    date_outliers = df['CompetitionSinceDate'] < '1960-01-01'  # Outlier: 1900-01-01
    df.loc[date_outliers, 'CompetitionSinceDate'] = pd.NaT
    
    return df