
import os
import numpy as np
from pandas.tseries.offsets import DateOffset


# Paths and file names
DATA_DIR = '../datasets/rossmann-store-sales'
STORE_FILE = os.path.join(DATA_DIR, 'store.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

LOG_DIR = "./artifacts"
LOG_FILE = os.path.join(LOG_DIR, "hypertuning.log")
STORAGE_URL = f"sqlite:///{os.path.join(LOG_DIR, 'hypertuning.db')}"

# Feature engineering settings
LAGS = [DateOffset(days=1),   DateOffset(days=2),
        DateOffset(weeks=1),  DateOffset(weeks=2),  DateOffset(weeks=3),
        DateOffset(months=1), DateOffset(months=3), DateOffset(months=6),
        DateOffset(years=1)]

DIFFS = [DateOffset(days=1),
         DateOffset(months=1), DateOffset(months=3), DateOffset(months=6)]

ROLL_WINDOWS = { 7: [DateOffset(days=1), DateOffset(weeks=1)],
                30: [DateOffset(months=1), DateOffset(months=3), DateOffset(months=6)]}

# Training settings
FORECAST_HORIZON = 42 # -> 6 weeks * 7 days/week
N_OUTER_SPLITS    = 2
OUTER_TRAIN_SIZE = 180  # days
N_INNER_SPLITS   = 2
INNER_TRAIN_SIZE = 90   # days
NUM_TRIALS = 5
LOG_PERIOD       = 20   # print xgb.cv metrics every LOG_PERIOD boosting rounds
SEED = 42
MONITOR_PERIODS = 100 # number of CV rounds to report in pruning callback (must be <= LOG_PERIOD)
NUM_STARTUP_TRIALS = 5 # Pruning is disabled until the given number of trials finish in the same study. After that, pruning is enabled for all subsequent trials.

EARLY_STOPPING_ROUNDS = 10
NUM_BOOST_ROUNDS = 1000
XGB_CONSTANTS: dict = {
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}


HYPERPARAMETERS = {
    "max_depth":        ("suggest_int",   2,    10,   {"log": False}),
    "learning_rate":    ("suggest_float", 1e-3, 0.3,  {"log": True}),
    "subsample":        ("suggest_float", 0.5,  1.0,  {"log": False}),
    "colsample_bytree": ("suggest_float", 0.5,  1.0,  {"log": False}),
    "min_child_weight": ("suggest_float", 1e-2, 50.0, {"log": True}),
    "reg_alpha":        ("suggest_float", 1e-8, 10.0, {"log": True}),
    "reg_lambda":       ("suggest_float", 1e-8, 10.0, {"log": True}),
    "gamma":            ("suggest_float", 0.0,  10.0, {"log": False}),
}


