from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pandas.tseries.offsets import DateOffset
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class PathsSettings(BaseModel):
    data_dir: str = "../datasets/rossmann-store-sales"
    log_dir: str = "./artifacts"
    store_file: str = ""
    train_file: str = ""
    log_file: str = ""
    storage_url: str = ""

    @model_validator(mode="after")
    def _compute_derived_paths(self) -> PathsSettings:
        if not self.store_file:
            self.store_file = os.path.join(self.data_dir, "store.csv")
        if not self.train_file:
            self.train_file = os.path.join(self.data_dir, "train.csv")
        if not self.log_file:
            self.log_file = os.path.join(self.log_dir, "hypertuning.log")
        if not self.storage_url:
            self.storage_url = f"sqlite:///{os.path.join(self.log_dir, 'hypertuning.db')}"
        return self


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class DiffOffsets(BaseModel):
    days: list[int]

    @field_validator("days")
    @classmethod
    def non_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("diffs.days must not be empty")
        return v

    def to_date_offsets(self) -> list[DateOffset]:
        return [DateOffset(days=d) for d in self.days]


class LagOffsets(BaseModel):
    days: list[int] = []
    weeks: list[int] = []
    months: list[int] = []
    years: list[int] = []

    @model_validator(mode="after")
    def _at_least_one_lag(self) -> LagOffsets:
        if not any([self.days, self.weeks, self.months, self.years]):
            raise ValueError("lags must define at least one offset")
        return self

    def to_date_offsets(self) -> list[DateOffset]:
        offsets: list[DateOffset] = []
        offsets += [DateOffset(days=d) for d in self.days]
        offsets += [DateOffset(weeks=w) for w in self.weeks]
        offsets += [DateOffset(months=m) for m in self.months]
        offsets += [DateOffset(years=y) for y in self.years]
        return offsets


class TrainingSettings(BaseModel):
    forecast_horizon: int
    roll_windows: list[str]
    diffs: DiffOffsets
    lags: LagOffsets

    @field_validator("forecast_horizon")
    @classmethod
    def positive_horizon(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("forecast_horizon must be positive")
        return v

    @field_validator("roll_windows")
    @classmethod
    def non_empty_windows(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("roll_windows must not be empty")
        return v


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class CVSettings(BaseModel):
    n_outer_splits: int
    n_inner_splits: int

    @field_validator("n_outer_splits", "n_inner_splits")
    @classmethod
    def positive_splits(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("split counts must be positive")
        return v


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

class HypertuningSettings(BaseModel):
    num_trials: int
    seed: int
    monitor_periods: int
    num_startup_trials: int
    num_jobs: int
    early_stopping_rounds: Optional[int] = None
    num_boost_rounds: int
    refit_val_fraction: float

    @field_validator("refit_val_fraction")
    @classmethod
    def valid_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("refit_val_fraction must be in the open interval (0, 1)")
        return v

    @field_validator("num_trials", "num_boost_rounds", "monitor_periods", "num_startup_trials")
    @classmethod
    def positive_counts(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v


# ---------------------------------------------------------------------------
# XGBoost constants
# ---------------------------------------------------------------------------

class XGBSettings(BaseModel):
    tree_method: str
    device: str
    objective: str
    eval_metric: str
    verbosity: int

    @field_validator("verbosity")
    @classmethod
    def valid_verbosity(cls, v: int) -> int:
        if v not in (0, 1, 2, 3):
            raise ValueError("verbosity must be 0, 1, 2, or 3")
        return v


# ---------------------------------------------------------------------------
# Hyperparameter search-space spec
# ---------------------------------------------------------------------------

_VALID_SUGGEST_METHODS = {"suggest_int", "suggest_float", "suggest_categorical"}


class HyperparameterSpec(BaseModel):
    method: str
    low: float
    high: float
    log: bool = False

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


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class AppSettings(BaseModel):
    paths: PathsSettings
    training: TrainingSettings
    cv: CVSettings
    hypertuning: HypertuningSettings
    xgb: XGBSettings
    hyperparameters: dict[str, HyperparameterSpec]

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppSettings:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
#
#   from pathlib import Path
#   from src.settings import AppSettings
#
#   settings = AppSettings.from_yaml(Path(__file__).parent.parent / "config.yaml")
#
#   # --- paths (derived paths are auto-computed) ---
#   print(settings.paths.store_file)        # ../datasets/rossmann-store-sales/store.csv
#   print(settings.paths.storage_url)       # sqlite:///./artifacts/hypertuning.db
#
#   # --- training ---
#   print(settings.training.forecast_horizon)           # 10
#   lags  = settings.training.lags.to_date_offsets()   # list[DateOffset]
#   diffs = settings.training.diffs.to_date_offsets()  # list[DateOffset]
#
#   # --- XGBoost params dict (ready to pass to xgb.train) ---
#   xgb_params = settings.xgb.model_dump()
#
#   # --- Optuna hyperparameter search space ---
#   for name, spec in settings.hyperparameters.items():
#       suggest = getattr(trial, spec.method)
#       low  = int(spec.low) if spec.method == "suggest_int" else spec.low
#       high = int(spec.high) if spec.method == "suggest_int" else spec.high
#       value = suggest(name, low, high, log=spec.log)
