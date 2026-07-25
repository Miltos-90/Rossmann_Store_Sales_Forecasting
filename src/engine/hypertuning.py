""" Nested time-series cross-validation with hyperopt using the native XGBoost API. """

import logging
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Any
from optuna.trial import Trial, TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import BaseCrossValidator
from optuna_integration.xgboost import XGBoostPruningCallback

from src.settings.models import HyperparameterSpec, AppSettings
from .callbacks import BoosterCollector
from .cv import TimeSeriesCV
from . import log, checkpoint as ckpt

logger = logging.getLogger(__name__)

_COMPLETE_STATES = [TrialState.PRUNED, TrialState.COMPLETE]  # States considered as completed for trial counting


def _suggest_params(trial: Trial, hyperparameters: dict[str, HyperparameterSpec]) -> dict[str, Any]:
    """Suggest hyperparameters for a trial using Optuna's sampling methods.

    Dynamically builds a hyperparameter dictionary based on the provided
    hyperparameters configuration, using the appropriate Optuna suggest method
    for each parameter.

    Args:
        trial: Active Optuna trial.
        hyperparameters: Search space definition mapping parameter names to
            HyperparameterSpec instances.

    Returns:
        Dictionary mapping hyperparameter names to suggested values.
    """
    params = {}
    for name, spec in hyperparameters.items():
        # Get the appropriate suggest method from the trial object
        suggest_fn = getattr(trial, spec.method)

        # Suggest a value for the hyperparameter using the specified method and bounds
        params[name] = suggest_fn(name, spec.low, spec.high, log=spec.log)

    return params


def _objective(
    trial: Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    study_name: str,
    config: AppSettings,
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
        config: Configuration dictionary for the study.

    Returns:
        Mean CV metric value (MAE or configured eval_metric) from the final boosting round.
    """
    metric = config.model_constants.eval_metric
    params = _suggest_params(trial, config.hyperparameters)

    # Run CV and pruning in a single call to XGBoost's cv function
    booster_collector = BoosterCollector()
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv=True, period=config.hypertuning.monitor_periods),
        xgb.callback.EarlyStopping(rounds=config.hypertuning.early_stopping_rounds),
        XGBoostPruningCallback(trial, observation_key=f"test-{metric}"),
        booster_collector,
    ]

    all_params = {**config.model_constants.model_dump(), **params}
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    num_boost_round = config.hypertuning.num_boost_rounds
    folds = list(cv.split(X))

    history = xgb.cv(params=all_params,
                     dtrain=dtrain,
                     num_boost_round=num_boost_round,
                     folds=folds,
                     metrics=[metric],
                     callbacks=callbacks,
                     verbose_eval=False,
                     seed=config.hypertuning.seed)

    log.trial(trial=trial, metric=metric, history=history)

    ckpt.trial(study_name=study_name,
               trial=trial,
               history=history,
               boosters=booster_collector.cvfolds,
               log_dir=config.path.output_dir)

    return history[f"test-{metric}-mean"].min()

def optimize(
        study_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        cv_settings: dict[str, Any],
        config: AppSettings) -> None:
    """ 
    Optimize hyperparameters using Optuna with nested cross-validation.

    Args:
        study_name (str): Name of the Optuna study.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        cv_settings (dict[str, Any]): Cross-validation settings.
        config (AppSettings): Configuration object for the study.

    Returns:
        None
    """

    pruner  = MedianPruner(n_startup_trials=config.hypertuning.num_startup_trials)
    sampler = TPESampler(seed=config.hypertuning.seed)    
    storage = ckpt.storage(log_dir=config.path.output_dir, study_name=study_name)
    study   = optuna.create_study(study_name=study_name,
                                  storage=storage,
                                  load_if_exists=True,
                                  direction="minimize",
                                  pruner=pruner,
                                  sampler=sampler)

    # Determine the number of remaining trials to run based on completed trials
    finished_trials  = study.get_trials(deepcopy=False, states=_COMPLETE_STATES)
    remaining_trials = config.hypertuning.num_trials - len(finished_trials)
    
    inner_cv = TimeSeriesCV(n_splits=config.cross_validation.n_inner_splits,
                            train_size=cv_settings["inner_train"],
                            test_size=cv_settings["inner_test"],
                            gap=config.horizon.days)

    obj_fcn = lambda trial: _objective(trial, X_train, y_train, inner_cv,
                                       study_name=study_name,
                                       config=config)

    study.optimize(obj_fcn,
                   n_trials=remaining_trials,
                   n_jobs=config.hypertuning.num_jobs,
                   timeout=config.hypertuning.timeout)
    log.study(study)
