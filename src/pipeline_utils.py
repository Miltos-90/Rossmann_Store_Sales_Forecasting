
""" Utility functions for the training pipeline. """

import os
import logging

import pandas as pd
import xgboost as xgb
import optuna

from typing import Tuple

from src.settings import PathSettings, AppSettings
from src.engine import TargetTransformer
from src.features import compute as compute_features
from src.engine import study_storage, refit as _refit

logger = logging.getLogger(__name__)


def load_data(path_config: PathSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Load sales and store data from CSV files specified in the configuration.
    Args:   
        path_config (PathSettings): The path settings containing file paths for sales and stores data.

    Returns:
        sales (pd.DataFrame): The sales data loaded from the CSV file.
        stores (pd.DataFrame): The store data loaded from the CSV file.
    """

    logger.info(f"Loading dataset.")
    sales  = pd.read_csv(path_config.train)
    stores = pd.read_csv(path_config.stores)

    return sales, stores


def generate_dataset(df: pd.DataFrame, config: AppSettings):
    """
    Generate features and target variable for model training.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing sales and store data.
        config (AppSettings): The application settings containing feature engineering and model configurations.
    
    Returns:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        trf (src.engine.TargetTransformer): The fitted target transformer.
    """

    logger.info(f"Generating features and target variable.")

    fh  = pd.DateOffset(days=-config.horizon.days)  # negative offset for forward difference
    trf = TargetTransformer(forecast_horizon=fh, anchor_col='lag_days_0')
    # The "anchor_col" is the column used to align the target variable with the features. 
    # In this case, we use 'lag_days_0' which represents the sales on the current day.
    # It will be generated in the feature engineering step.

    y = df.set_index(['Store', 'Date'])['Sales']
    X = (df
        .set_index(['Date'])
        .groupby('Store')
        .apply(lambda df: compute_features(df, config.feature_engineering, config.horizon)))

    trf.fit(X)
    y = trf.transform(y)
    X = X.loc[y.index]  # align features with target

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')

    return X, y, trf


def refit(X_train: pd.DataFrame, y_train: pd.Series, study_name: str, config: AppSettings) -> xgb.Booster:
    """ 
    Refit the best model for the given outer fold or load it if it already exists.

    Args:
        X_train (pd.DataFrame): The feature matrix for the outer-fold training data.
        y_train (pd.Series): The target variable for the outer-fold training data.
        study_name (str): The name of the Optuna study corresponding to the outer fold.
        config (AppSettings): The application settings containing model configurations.

    Returns:
        xgb.Booster: The fitted XGBoost model for the outer fold.
    """

    # If the model for this outer fold already exists, load it; otherwise, refit the best model and save it.
    booster_name = f"{study_name}_best_model.ubj"
    booster_path = os.path.join(config.path.output_dir, booster_name)  # Path to save the best model for this outer fold.

    if os.path.exists(booster_path):
        # Load the existing model
        logger.info(f"Best model for '{study_name}' already exists. Loading from {booster_path}.")
        
        booster = xgb.Booster()
        booster.load_model(booster_path)
    else:
        # Refit the best model using the best hyperparameters from the study
        logger.info(f"Refitting best model for '{study_name}'.")

        storage = study_storage(log_dir=config.path.output_dir, study_name=study_name)
        study   = optuna.load_study(study_name=study_name, storage=storage)
        booster = _refit(X_train, y_train, study.best_trial, config.model_constants)
        booster.save_model(booster_path)
        logger.info(f"Best model for '{study_name}' saved to {booster_path}.")

    return booster