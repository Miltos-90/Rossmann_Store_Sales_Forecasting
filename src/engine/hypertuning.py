"""Nested time-series cross-validation with hyperopt using the native XGBoost API."""

import logging
import os
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Any
from sklearn.model_selection import BaseCrossValidator
from optuna.trial import Trial, TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration.xgboost import XGBoostPruningCallback

from .callbacks import BoosterCollector
from .cv import TimeSeriesCV
from . import utils


logger = logging.getLogger(__name__)

def _suggest_params(trial: Trial, hyperparameters: dict) -> dict[str, Any]:
    """Suggest hyperparameters for a trial using Optuna's sampling methods.

    Dynamically builds a hyperparameter dictionary based on the provided
    hyperparameters configuration, using the appropriate Optuna suggest method
    for each parameter.

    Args:
        trial: Active Optuna trial.
        hyperparameters: Search space definition mapping parameter names to
            (method, low, high, kwargs) tuples.

    Returns:
        Dictionary mapping hyperparameter names to suggested values.
    """
    params = {}
    for name, (method, low, high, kwargs) in hyperparameters.items():
        suggest_fn = getattr(trial, method)
        params[name] = suggest_fn(name, low, high, **kwargs)
    return params


def _objective(
    trial: Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    study_name: str,
    study_config: dict,
) -> float:
    """Optuna objective for broad search over the full hyperparameter space.

    Suggests values from wide ranges for all XGBoost hyperparameters, performs
    nested time-series cross-validation, and records per-fold models and metrics.
    Integrates per-fold pruning via XGBoostPruningCallback for early termination
    of unpromising trials.

    Args:
        trial: Active Optuna trial.
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        cv: Time-series cross-validation splitter.
        study_name: Name of the Optuna study, used for organizing artifacts.
        study_config: Optuna study settings (see nested_cv for key descriptions).

    Returns:
        Mean CV metric value (MAE or configured eval_metric) from the final boosting round.
    """
    metric = study_config["xgb_constants"]["eval_metric"]
    params = _suggest_params(trial, study_config["hyperparameters"])

    # Run CV and pruning in a single call to XGBoost's cv function
    booster_collector = BoosterCollector()
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv=True, period=study_config["monitor_periods"]),
        xgb.callback.EarlyStopping(rounds=study_config["early_stopping_rounds"]),
        XGBoostPruningCallback(trial, observation_key=f"test-{metric}"),
        # The observation_key in the pruning callback reads from XGBoost's internal evals_log
        # dict (keyed as "test-<metric>") - not from the final DataFrame column names ("test-<metric>-mean").
        booster_collector,
    ]

    all_params = {**study_config["xgb_constants"], **params}
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    num_boost_round = study_config["num_boost_rounds"]
    folds = list(cv.split(X))

    history = xgb.cv(params=all_params,
                     dtrain=dtrain,
                     num_boost_round=num_boost_round,
                     folds=folds,
                     metrics=[metric],
                     callbacks=callbacks,
                     verbose_eval=False,
                     verbose=False,
                     seed=study_config["seed"])

    utils.log_trial_cv_results(trial=trial, metric=metric, history=history)

    utils.save_trial_artifacts(study_name=study_name,
                               trial=trial,
                               history=history,
                               boosters=booster_collector.cvfolds,
                               log_dir=study_config["log_dir"])

    return history[f"test-{metric}-mean"].min()


def _wide_to_long_predictions(s: pd.DataFrame) -> pd.Series:
    """ Convert wide-format predictions with MultiIndex (Date, Store) and columns for each day ahead
        to long-format predictions with MultiIndex (Date, Store, Forecast Date) and a single value column.

        Args:
            s: DataFrame with MultiIndex (Date, Store) and columns for each day ahead

        Returns:
            Series with MultiIndex (Date, Store, Forecast Date) and values containing the predictions.
    """

    # Reset index to make Date and Store regular columns
    s_long = s.reset_index()

    # Melt from wide to long format
    s_long = s_long.melt(id_vars=["Date", "Store"],
                         var_name="lead_column",
                         value_name="value")

    # Extract days ahead from column name (e.g., "lead_1_days" -> 1)
    s_long['days_ahead'] = (s_long['lead_column'].astype(int) + 1)

    # Calculate forecast date by adding days to the base Date
    s_long['Forecast Date'] = s_long['Date'] + pd.to_timedelta(s_long['days_ahead'], unit='D')

    # Create the final long-format dataframe with MultiIndex (Date, Store, Forecast Date)
    s_long = (s_long
              .set_index(['Date', 'Store', 'Forecast Date'])
              .drop(['lead_column', 'days_ahead'], axis=1)
              ['value']  # Implicitly convert to Series
              .sort_index())

    return s_long


def predict(booster: xgb.Booster, X_test: pd.DataFrame) -> pd.Series:
    """ 
    Generate predictions for the given test data using the provided XGBoost booster.

    Args:
        booster: Trained XGBoost booster.
        X_test: Test features.

    Returns:
        Series with predictions.
    """
    test_dmatrix  = xgb.DMatrix(X_test, enable_categorical=True)
    log_preds_raw = booster.predict(test_dmatrix)
    log_preds = pd.DataFrame(data=log_preds_raw, index=X_test.index)
    preds_long = _wide_to_long_predictions(log_preds)
    preds = utils.overwrite_closed_sales(preds_long, mode='zero', index_name="Forecast Date")
    preds = preds.apply(np.expm1)

    return preds

def refit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_trial: optuna.trial.FrozenTrial,
    config: dict[str, Any],
    val_fraction: float,
) -> xgb.Booster:
    """ 
    Refit XGBoost model on the entire outer fold training set using the best hyperparameters found in the inner loop.

    A temporal validation split (last val_fraction of the training rows) is used
    with early stopping so that the optimal boosting rounds re-calibrate to the
    larger outer-fold dataset rather than being fixed at the inner-CV value.

    Args:
        X_train: Training features for the outer fold.
        y_train: Training targets for the outer fold.
        best_trial: Optuna trial object containing the best hyperparameters from the inner loop.
        config: Dictionary containing either the flat XGBoost constants dict directly,
            or a study config dict with an "xgb_constants" key.
        val_fraction: Fraction of X_train (taken from the end, preserving temporal order)
            to use as a validation set for early stopping.

    Returns:
        Trained XGBoost booster fitted on the entire outer fold training set with the best hyperparameters.
    """
    xgb_constants = config["xgb_constants"]
    num_boost_round = config.get("num_boost_rounds")
    early_stopping_rounds = config["early_stopping_rounds"]
    params = {**xgb_constants, **best_trial.params}
    
    # Temporal validation split for early stopping
    n_val = max(1, int(len(X_train) * val_fraction))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    dtrain_sub = xgb.DMatrix(X_tr,  label=y_tr,  enable_categorical=True)
    dval       = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    # Phase 1: find optimal round count via early stopping on the held-out val set
    booster = xgb.train(
        params=params,
        num_boost_round=num_boost_round,
        dtrain=dtrain_sub,
        evals=[(dval, "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=early_stopping_rounds)],
        verbose_eval=False,
    )

    return booster


def optimize(study_name: str, X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> None:
    """ 
    Optimize hyperparameters using Optuna with nested cross-validation.

    Args:
        study_name (str): Name of the Optuna study.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        config (dict): Configuration dictionary for the study.

    Returns:
        None
    """

    inner_cv = TimeSeriesCV(n_splits=config["n_inner_splits"],
                            horizon=config["forecast_horizon"],
                            train_size=config["inner_train_size"])
    
    pruner = MedianPruner(n_startup_trials=config["n_startup_trials"])

    sampler = TPESampler(seed=config["seed"])

    obj_fcn = lambda trial: _objective(trial, X_train, y_train, inner_cv,
                                       study_name=study_name,
                                       study_config=config)

    study = optuna.create_study(study_name=study_name,
                                storage=config["storage_url"],
                                load_if_exists=True,
                                direction="minimize",
                                pruner=pruner,
                                sampler=sampler)

    finished_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED, TrialState.COMPLETE])
    remaining_trials = config["n_trials"] - len(finished_trials)

    study.optimize(obj_fcn, n_trials=remaining_trials, n_jobs=config["n_jobs"])
    utils.log_study(study)

    return