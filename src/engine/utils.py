""" Utility functions for training and evaluation pipelines """

import logging
import optuna
import pandas as pd
import xgboost as xgb

from engine.target_transformer import TargetTransformer
from settings import CVSettings, XGBSettings

logger = logging.getLogger(__name__)


def compute_cv_sizes(
    total_days: int,
    forecast_horizon: int,
    cv_config: CVSettings,
) -> dict:
    """
    Compute sliding-window sizes for nested time-series CV.

    Formulas (gap = forecast_horizon = H, test sizes fixed to H):
        outer_train_size = N - H * (K_out + 1)
        inner_train_size = N - H * (K_out + K_in + 2)
        outer_test_size  = H
        inner_test_size  = H

    Validity: N > h * (K_out + K_in + 2)

    Args: 
        total_days: Total number of days in the dataset for one store (N).
        forecast_horizon: Forecast horizon in days (H).
        cv_config: Cross-validation settings containing the number of outer and inner splits.
    """

    # Rename variables for clarity in formulas
    n     = total_days
    h     = forecast_horizon
    k_out = cv_config.n_outer_splits
    k_in  = cv_config.n_inner_splits

    outer_train_size = n - h * (k_out + 1)
    inner_train_size = n - h * (k_out + k_in + 2)

    min_days = h * (k_out + k_in + 2) + 1
    if inner_train_size <= 0:
        raise ValueError(
            f"Dataset too small: {n} days available, "
            f"at least {min_days} required for a 1-day inner training window."
        )

    config = {
        "outer_train": outer_train_size,
        "inner_train": inner_train_size,
        "outer_test":  h,
        "inner_test":  h,
    }

    return config


def predict(X: pd.DataFrame, booster: xgb.Booster, transformer: TargetTransformer):
    """ 
    Predict sales using a trained XGBoost booster and inverse transform the predictions.

    Args:
        X (pd.DataFrame): Feature matrix for prediction.
        booster (xgb.Booster): Trained XGBoost booster.
        transformer (TargetTransformer): Transformer to inverse transform the predictions.

    Returns:
        pd.Series: Predicted sales values.
    """
    logger.info(f"Predicting sales for {X.shape[0]} samples.")

    test_dmatrix = xgb.DMatrix(X, enable_categorical=True)
    preds_raw = booster.predict(test_dmatrix)
    preds = pd.Series(data=preds_raw, index=X.index, name='Sales')
    preds = transformer.inverse_transform(preds)

    return preds


def refit(
    X: pd.DataFrame,
    y: pd.Series,
    best_trial: optuna.trial.FrozenTrial,
    config: XGBSettings,
) -> xgb.Booster:
    """ 
    Refit XGBoost model on the entire outer fold training set using the best hyperparameters found in the inner loop.

    A temporal validation split (last val_fraction of the training rows) is used
    with early stopping so that the optimal boosting rounds re-calibrate to the
    larger outer-fold dataset rather than being fixed at the inner-CV value.

    Args:
        X: Training features for the outer fold.
        y: Training targets for the outer fold.
        best_trial: Optuna trial object containing the best hyperparameters from the inner loop.
        config: XGBSettings object containing the XGBoost constants.

    Returns:
        Trained XGBoost booster fitted on the entire outer fold training set with the best hyperparameters.
    """
    logger.debug(f"Refitting with hyperparameters: {best_trial.params}")
    num_boost_round = best_trial.user_attrs['best_n_rounds']
    params  = {**config.model_dump(), **best_trial.params}
    dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)
    booster = xgb.train(params=params,
                        num_boost_round=num_boost_round,
                        dtrain=dmatrix)

    return booster
