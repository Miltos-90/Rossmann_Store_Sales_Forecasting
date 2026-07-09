"""
This module defines the configuration settings.
"""

import os
import yaml
import pandas as pd

from pathlib import Path
from typing import List, Any
from typing_extensions import Annotated
from pydantic import (
    BaseModel, ConfigDict, AfterValidator, BeforeValidator,
    field_validator, model_validator
)

from . import validators


# Define valid hyperparameter suggestion methods
_VALID_SUGGEST_METHODS = {"suggest_int", "suggest_float", "suggest_categorical"}



# ---------- Annotated Types ----------
PositiveInt   = Annotated[int, AfterValidator(validators.positive_int)]
Offset        = Annotated[pd.DateOffset, AfterValidator(validators.positive_offset)]
OffsetList    = Annotated[List[Offset], BeforeValidator(validators.offsets)]
HorizonOffset = Annotated[
    pd.DateOffset,
    AfterValidator(validators.positive_offset),       # Runs 2nd
    BeforeValidator(validators.int_to_offset("days")) # Runs 1st
]

# ---------- Models ----------
class PathSettings(BaseModel):
    data_dir: Path
    log_dir:  Path
    stores:   Path
    train:    Path
    logs:     Path
    predictions: Path

    @model_validator(mode="after")
    def _compute_derived_paths(self) -> PathSettings:
        """ 
        Compute the full paths for stores, train, logs, and predictions based on the data_dir and log_dir. 
        
        Returns:
            PathSettings: The instance with updated paths.
        """
        self.stores      = os.path.join(self.data_dir, self.stores)
        self.train       = os.path.join(self.data_dir, self.train)
        self.logs        = os.path.join(self.log_dir, self.logs)
        self.predictions = os.path.join(self.log_dir, self.predictions)

        return self


    # Convert all relative paths to absolute paths
    @model_validator(mode="after")  
    def _convert_paths_to_absolute(self) -> PathSettings:
        """ 
        Convert all relative paths to absolute paths.
        """
        self.data_dir    = os.path.abspath(self.data_dir)
        self.log_dir     = os.path.abspath(self.log_dir)
        self.stores      = os.path.abspath(self.stores)
        self.train       = os.path.abspath(self.train)
        self.logs        = os.path.abspath(self.logs)
        self.predictions = os.path.abspath(self.predictions)
        return self


class FeatureEngineeringSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    windows:  list[str]
    diffs:    OffsetList
    lags:     OffsetList
    holidays: dict[str, Any]


class CVSettings(BaseModel):
    n_outer_splits: PositiveInt
    n_inner_splits: PositiveInt


class HypertuningSettings(BaseModel):
    num_trials:             PositiveInt
    seed:                   PositiveInt
    monitor_periods:        PositiveInt
    num_startup_trials:     PositiveInt
    num_jobs:               int  # -1 means use all available cores (joblib/Optuna convention)

    @field_validator("num_jobs")
    @classmethod
    def valid_num_jobs(cls, v: int) -> int:
        if v != -1 and v <= 0:
            raise ValueError("num_jobs must be a positive integer or -1 (all cores).")
        return v
    early_stopping_rounds:  PositiveInt
    num_boost_rounds:       PositiveInt
    refit_val_fraction:     float

    @field_validator("refit_val_fraction")
    @classmethod
    def valid_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("refit_val_fraction must be in the open interval (0, 1)")
        return v


class XGBSettings(BaseModel):
    tree_method: str
    device:      str
    objective:   str
    eval_metric: str
    verbosity:   int

    @field_validator("verbosity")
    @classmethod
    def valid_verbosity(cls, v: int) -> int:
        if v not in (0, 1, 2, 3):
            raise ValueError("verbosity must be 0, 1, 2, or 3")
        return v


class HyperparameterSpec(BaseModel):
    method: str
    low:    float
    high:   float
    log:    bool # Whether to sample in log space

    @field_validator("method")
    @classmethod
    def valid_method(cls, v: str) -> str:
        if v not in _VALID_SUGGEST_METHODS:
            raise ValueError(f"method must be one of {_VALID_SUGGEST_METHODS}")
        return v

    @model_validator(mode="after")
    def low_less_than_high(self) -> HyperparameterSpec:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be strictly less than high ({self.high})")
        return self


class AppSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    horizon:             HorizonOffset
    path:                PathSettings
    cross_validation:    CVSettings
    feature_engineering: FeatureEngineeringSettings
    hypertuning:         HypertuningSettings
    model_constants:     XGBSettings
    hyperparameters:     dict[str, HyperparameterSpec]

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppSettings:
        """
        Load settings from a YAML file and return an instance of AppSettings.
        Args:
            path (str | Path): The path to the YAML configuration file.
        Returns:
            AppSettings: An instance of AppSettings populated with the data from the YAML file.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)
