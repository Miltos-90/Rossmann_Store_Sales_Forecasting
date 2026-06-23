import pandas as pd
import numpy as np

def _date_from_week_year(row: pd.Series, year_col: str, week_col: str, day: int = 1) -> pd.Series:
    """
    Convert ISO week and year values into a datetime object.

    Parameters
    ----------
    row : pd.Series
        A single row from a DataFrame, typically passed when using `DataFrame.apply(axis=1)`.
    year_col : str
        The column name in `row` containing the year value.
    week_col : str
        The column name in `row` containing the ISO week number.
    day : int, default=1
        The ISO weekday number to use when constructing the date
        (`1` = Monday, `7` = Sunday).

    Returns
    -------
    pd.Timestamp or pd.NaT
        The corresponding date for the given (year, week, day) combination,
        or `NaT` if either year or week is missing.
    """

    if pd.isna(row[year_col]) or pd.isna(row[week_col]):
        out = pd.NaT  # Missing date

    else:
        date_str = f"{int(row[year_col])}-W{int(row[week_col]):02d}-{day}"
        out = pd.to_datetime(date_str, format="%G-W%V-%u")

    return out


def process_store_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich store metadata by converting year/week or year/month fields into datetime columns.

    It then removes the original component columns and replaces implausible or invalid
    competition dates (e.g., earlier than 1960) with `NaT`.

    Parameters
    ----------
    df : pd.DataFrame
        Store-level DataFrame containing information about promotions and competition.
        Expected columns include:
        - `'Promo2SinceYear'`, `'Promo2SinceWeek'`
        - `'CompetitionOpenSinceYear'`, `'CompetitionOpenSinceMonth'`

    Returns
    -------
    pd.DataFrame
        The cleaned and enriched DataFrame, containing:
        - `'Promo2SinceDate'` : datetime of Promo2 start (first day of that ISO week)
        - `'CompetitionSinceDate'` : datetime of competition opening (first day of that month)
        The original year/week/month columns are removed.
    """

    args = ('Promo2SinceYear', 'Promo2SinceWeek')
    df['Promo2SinceDate'] = df.apply(_date_from_week_year, args=args, axis=1)

    args = ('CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth')
    df['CompetitionSinceDate'] = df.apply(_date_from_week_year, args=args, axis=1)

    remove_cols = ['Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'] # not needed anymore
    df.drop(remove_cols, axis=1, inplace=True)

    date_outliers = df['CompetitionSinceDate'] < '1960-01-01'  # Outlier: 1900-01-01
    df.loc[date_outliers, 'CompetitionSinceDate'] = pd.NaT

    return df


def drop_null_targets(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Drop samples with null target values since they cannot be used for training or evaluation.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target series aligned with X

    Returns:
        tuple[pd.DataFrame, pd.Series]: Filtered X and y with null target samples removed
    """
    non_null_targets = ~y.isnull()
    X, y = X[non_null_targets], y[non_null_targets]
    return X, y


def preprocess_data(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the sales and store data for modeling.

    Args:
        sales (pd.DataFrame): The sales data.
        stores (pd.DataFrame): The store data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for feature engineering.
    """

    # Compute the Promo2SinceDate and CompetitionStartDate columns from year/week and year/month
    stores['Promo2SinceDate'] = stores.apply(_date_from_week_year,
                                             args=('Promo2SinceYear', 'Promo2SinceWeek'),
                                             axis=1)

    stores['CompetitionStartDate'] = stores.apply(_date_from_week_year,
                                                  args=('CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'),
                                                  axis=1)

    df = sales.merge(stores, on='Store', how='inner')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Store', 'Date']) # Sort by Store and Date to ensure chronological order for feature engineering 

    # Normalize the StateHoliday column to a boolean indicator for holidays and create a separate column for school holidays
    no_state_holiday = df['StateHoliday'].isin(['0', 0]) | df['StateHoliday'].isna()
    df.loc[no_state_holiday, 'StateHoliday'] = 'NoHoliday'

    # Create boolean columns for state and school holidays
    df['isStateHoliday'] = df['StateHoliday'] != 'NoHoliday'
    df['isSchoolHoliday'] = df['SchoolHoliday'].astype(bool)

    # Drop the original StateHoliday and SchoolHoliday columns since we now have boolean indicators
    drop_cols = ['Promo2SinceYear', 'Promo2SinceWeek', 'StateHoliday', 'SchoolHoliday',
                 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']

    df.drop(columns=drop_cols, inplace=True)

    return df


def _is_closed_day(dates: pd.Series) -> pd.Series:
    """
    Determines if a given date is a closed day (Sunday) for the store.
    
    Parameters:
    dates (pd.Series): The dates to check.

    
    Returns:
    bool: True if the date is a closed day, False otherwise.
    """
    return dates.dt.day_name() == 'Sunday'


def interpolate_closed_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the 'Sales' values for closed days (Sundays) in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing at least 'Date', 'Store', and 'Sales' columns.
    
    Returns:
    pd.DataFrame: DataFrame with interpolated 'Sales' values for closed days.
    """

    df.loc[_is_closed_day(df['Date']), 'Sales'] = np.nan

    # 2. We sort by Date inside the groupby to ensure linear interpolation happens chronologically
    df['Sales'] = (df
                   .groupby('Store', group_keys=False)
                   .apply(lambda group: group.sort_values('Date').interpolate(method='linear'))
                   ['Sales'] # Isolate the interpolated sales
                   .round(0) # Round to nearest integer since sales are whole numbers
                   .astype('int'))

    return df


def set_closed_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrites the 'Sales' values for closed days (Sundays) in the DataFrame with zeros.

    Parameters:
    df (pd.DataFrame): DataFrame containing at least 'Date' and 'Sales' columns.

    Returns:
    pd.DataFrame: DataFrame with 'Sales' values set to zero for closed days.
    """
    df.loc[_is_closed_day(df['Date']), 'Sales'] = 0
    df['Sales'] = df['Sales'].round(0).astype('int') # Ensure Sales is integer type after setting to zero

    return df
