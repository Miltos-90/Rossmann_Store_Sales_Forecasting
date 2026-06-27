
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

# Training settings
LAGS = (
    [DateOffset(days=d) for d in np.arange(1, 7)] + 
    [DateOffset(weeks=w) for w in np.arange(1, 4)] + 
    [DateOffset(months=m) for m in np.arange(1, 4)] +
    [DateOffset(years=1)]
)

DIFFS = [DateOffset(days=d) for d in np.arange(1, 7)]

ROLL_WINDOWS = { 7: [DateOffset(days=1)], 30: [DateOffset(days=1)], 90: [DateOffset(days=1)]}


FORECAST_HORIZON = 10 # of days ahead to predict

# CV settings
N_OUTER_SPLITS   = 5
N_INNER_SPLITS   = 3

# Hyperparameter tuning settings
NUM_TRIALS              = 5#100 # number of Optuna trials for hyperparameter tuning in each outer CV split
SEED                    = 42 # random seed for reproducibility
MONITOR_PERIODS         = 100 # number of CV rounds to report in pruning callback
NUM_STARTUP_TRIALS      = 3#5 # Pruning is disabled until the given number of trials finish in the same study. After that, pruning is enabled for all subsequent trials.
NUM_JOBS                = -1 # number of parallel jobs for Optuna.  Set to -1 to use all available cores.
EARLY_STOPPING_ROUNDS   = 10 # Number of rounds with no improvement after which training will be stopped.  Set to None to disable early stopping.
NUM_BOOST_ROUNDS        = 10000 # Maximum number of boosting rounds to train.  Early stopping may cause training to stop before this number is reached.
REFIT_VAL_FRACTION      = 0.1 # Fraction of the training data to set aside for early stopping when refitting the final model with the best hyperparameters on the entire outer fold training set.

XGB_CONSTANTS: dict = {
    "tree_method": "hist",
    "device": "cpu",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}

HYPERPARAMETERS = {
    "max_depth":        ("suggest_int",   3,    10,   {"log": False}),
    "learning_rate":    ("suggest_float", 0.001, 0.3, {"log": True}),
    "subsample":        ("suggest_float", 0.6,  1.0,  {"log": False}), 
    "colsample_bytree": ("suggest_float", 0.6,  1.0,  {"log": False}), 
    "min_child_weight": ("suggest_int",   1,    20,   {"log": False}),
    "reg_alpha":        ("suggest_float", 1e-3, 1.0,  {"log": True}),
    "reg_lambda":       ("suggest_float", 1e-3, 5.0,  {"log": True}),
    "gamma":            ("suggest_float", 0.0,  2.0,  {"log": False}),
}


