"""
This module defines the validators for the configuration settings.
"""

from mimetypes import add_type
from typing import Any, List

import pandas as pd

# Set of valid keys for pd.DateOffset
_VALID_OFFSET_KEYS = {"days", "weeks", "months", "years"}


def int_to_offset(unit: str):
    """
    Factory function to create a validator that converts an integer to a pd.DateOffset of the specified unit.
    
    Args:
        unit (str): The unit of time for the offset (e.g., "days", "weeks", "months", "years").
    """

    def validator(v: Any) -> pd.DateOffset:
        """
        Validator function that converts an integer to a pd.DateOffset of the specified unit.
        """
        if v is None:
            return None
        if isinstance(v, int):
            return pd.DateOffset(**{unit: v})
        if isinstance(v, pd.DateOffset):
            return v
        # At this point we can raise an error.
        raise TypeError(f"Value must be an int or pd.DateOffset, got {type(v)} instead.")
    
    return validator

def positive_offset(v: pd.DateOffset) -> pd.DateOffset:
    """
    Ensures that the provided pd.DateOffset has non-negative values for all its components.
    Raises a ValueError if any component is negative.
    """
    for kwd, val in v.kwds.items():
        if val < 0:
            raise ValueError(f"The offset value for '{kwd}' cannot be negative.")
    return v


def positive_int(v: int) -> int:
    """
    Ensures that the provided integer is positive.
    Raises a ValueError if the integer is not positive.
    """
    if v <= 0:
        raise ValueError("Value must be a positive integer.")
    return v


def offsets(data: Any) -> List[pd.DateOffset]:
    """
    Intercepts incoming dictionary of {days: [], weeks: []...} 
    and flattens it directly into a single List[pd.DateOffset]
    """
    if not isinstance(data, dict):
        return data
    
    for key in data.keys():
        if key not in _VALID_OFFSET_KEYS:
            raise ValueError(f"Invalid offset key '{key}'. Valid keys are: {_VALID_OFFSET_KEYS}")

    add_to_list = lambda key: key in data and isinstance(data[key], list) and data[key]

    flattened = []
    for key in ["days", "weeks", "months", "years"]:
        if add_to_list(key):
            validator = int_to_offset(key)
            flattened.extend([validator(v) for v in data[key]])

    return flattened