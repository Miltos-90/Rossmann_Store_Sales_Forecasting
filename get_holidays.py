
# %%

import pandas as pd
import holidays

from src.preprocessing import preprocess_data
from src.settings import AppSettings


COUNTRY = "DE"
LANGUAGE = "en_US"  # language for holiday names returned by the package


def _get_holidays_for_subdivision(
        country: str, language: str, subdiv: str, years: list[int]
    ) -> pd.DataFrame:
    """ 
    Get holidays for a specific subdivision of a country for the given years.

    Args:
        country (str): The country code (e.g., "DE" for Germany).
        language (str): The language for holiday names.
        subdiv (str): The subdivision code (e.g., state or region code).
        years (list[int]): The list of years for which to retrieve holidays.
    
    """
    
    # Dictionary of holidays for the given country, subdivision, and years.
    # keys are dates (datetime.date), values are holiday names (str)
    subdiv_holiday_dict = holidays.country_holidays(country = country, 
                                                    language=language,
                                                    subdiv=subdiv, 
                                                    years=years)
    
    subdiv_holiday_df = pd.DataFrame.from_dict(subdiv_holiday_dict,
                                            orient='index', 
                                            columns=['name'])
    
    subdiv_holiday_df["subdiv"] = subdiv  # Add a column for the subdivision code

    subdiv_holiday_df = (subdiv_holiday_df.reset_index()
                         .rename(columns={'index': 'Date'}))

    return subdiv_holiday_df


def get_holidays(
        years: list[int],
        country: str = COUNTRY,
        language: str = LANGUAGE) -> pd.DataFrame:
    """
    Get holidays for all subdivisions of a country for the given years.

    Args:
        years (list[int]): The list of years for which to retrieve holidays.
        country (str): The country code (e.g., "DE" for Germany).
        language (str): The language for holiday names.
        
    Returns:
        pd.DataFrame: A DataFrame containing holidays for all subdivisions.
                      Columns include 'Date', 'name', and 'subdiv'.
    """
    
    # Get all subdivision codes for the specified country
    subdivision_codes = list(holidays.country_holidays(country=country,
                                                       years=years).subdivisions) 

    subdiv_holiday_df = []
    for subdiv in subdivision_codes:
        subdiv_holidays = _get_holidays_for_subdivision(country, language, subdiv, years)
        subdiv_holiday_df.append(subdiv_holidays)

    subdiv_holiday_df = pd.concat(subdiv_holiday_df)

    return subdiv_holiday_df



# %%

config = AppSettings.from_yaml('./config.yaml')

sales  = pd.read_csv(config.path.train)
stores = pd.read_csv(config.path.stores)
df     = preprocess_data(sales, stores)


holiday_df= df[['Date', 'Store', 'isStateHoliday', 'isSchoolHoliday']]

holiday_df.head()


# %%


years_in_dset = holiday_df['Date'].dt.year.unique().tolist()
public_holidays = get_holidays(years_in_dset)

public_holidays



# %%

# We 
