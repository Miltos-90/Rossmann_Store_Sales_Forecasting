
import os
import numpy as np
from hyperopt import hp
from pandas.tseries.offsets import DateOffset


# Paths and file names
DATA_DIR = '../datasets/rossmann-store-sales'
STORE_FILE = os.path.join(DATA_DIR, 'store.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

LOG_DIR = "./artifacts"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
LOG_FILE = os.path.join(LOG_DIR, "hypertuning.log")

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
NUM_TRIALS       = 2
LOG_PERIOD       = 20   # log CV metrics every LOG_PERIOD boosting rounds

XGB_CONSTANTS: dict = {
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
    "early_stopping_rounds": 10,
}

SEARCH_SPACE: dict = {
    "num_boost_round":  hp.choice("num_boost_round",   [100, 200, 300, 500]),
    "max_depth":        hp.choice("max_depth",         [3, 4, 5, 6, 8]),
    "eta":              hp.loguniform("eta",           np.log(0.01), np.log(0.3)),
    "subsample":        hp.uniform("subsample",        0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "min_child_weight": hp.choice("min_child_weight",  [1, 3, 5, 10, 20]),
    "alpha":            hp.loguniform("alpha",         np.log(1e-4), np.log(10.0)),
    "lambda":           hp.loguniform("lambda",        np.log(1e-4), np.log(10.0)),
}
