""" Training pipeline. """
import os
import logging
import warnings
import optuna
import pandas as pd

from typing import Tuple
from sklearn.metrics import mean_absolute_error

import src

# Read configuration from YAML file
config = src.AppSettings.from_yaml('./config.yaml')

# Set up logging
os.makedirs(config.path.log_dir, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    filename=config.path.logs,
                    format="%(levelname)s  %(message)s")

logger = logging.getLogger(__name__)

# TODO: Use same functions in the notebook as well.
# TODO: Argparse for command line arguments to specify config file path, etc.
# TODO: Readme

def load_data(path_config: src.settings.PathSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Load sales and store data from CSV files specified in the configuration.
    Args:   
        path_config (src.settings.PathSettings): The path settings containing file paths for sales and stores data.

    Returns:
        sales (pd.DataFrame): The sales data loaded from the CSV file.
        stores (pd.DataFrame): The store data loaded from the CSV file.
    """

    logger.info(f"Loading dataset.")
    sales  = pd.read_csv(path_config.train)
    stores = pd.read_csv(path_config.stores)

    return sales, stores


def generate_dataset(df: pd.DataFrame, config: src.AppSettings):
    """
    Generate features and target variable for model training.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing sales and store data.
        config (src.AppSettings): The application settings containing feature engineering and model configurations.
    
    Returns:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        trf (src.engine.TargetTransformer): The fitted target transformer.
    """

    logger.info(f"Generating features and target variable.")

    fh  = pd.DateOffset(days=-config.horizon.days)  # negative offset for forward difference
    trf = src.engine.TargetTransformer(forecast_horizon=fh, anchor_col='lag_days_0')
    # The "anchor_col" is the column used to align the target variable with the features. 
    # In this case, we use 'lag_days_0' which represents the sales on the current day.
    # It will be generated in the feature engineering step.

    y = df.set_index(['Store', 'Date'])['Sales']
    X = (df
        .set_index(['Date'])
        .groupby('Store')
        .apply(lambda df: src.features.compute(df, config.feature_engineering, config.horizon)))

    trf.fit(X)
    y = trf.transform(y)
    X = X.loc[y.index]  # align features with target

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')

    return X, y, trf


if __name__ == "__main__":

    sales, stores = load_data(config.path)

    ##################################################
    stores_to_use = [1, 2, 3]
    stores = stores[stores['Store'].isin(stores_to_use)]
    ##################################################


    df = src.preprocessing.preprocess_data(sales, stores)
    X, y, trf = generate_dataset(df, config)

    # Run nested cross-validation
    num_days = X.index.get_level_values("Date").nunique()
    cv_sizes = src.engine.compute_cv_sizes(num_days,
                                    config.horizon.days,
                                    config.cross_validation)

    outer_cv = src.engine.TimeSeriesCV(n_splits=config.cross_validation.n_outer_splits,
                                       gap=config.horizon.days,
                                       train_size=cv_sizes['outer_train'],
                                       test_size=cv_sizes['outer_test'])

    preds = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):

        logger.info(f"Outer fold {fold}/{config.cross_validation.n_outer_splits}")

        # Split the data into training and test sets for the current outer fold
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        # Hyperparameter tuning on the outer-fold training data (embargoed inner CV inside optimize).
        study_name = f"outer_fold_{fold}"
        src.engine.optimize(study_name=study_name,
                            X_train=X_train,
                            y_train=y_train,
                            cv_settings=cv_sizes,
                            config=config)

        # Refit on the outer-fold training data using the best hyperparameters.
        storage = src.engine.study_storage(log_dir=config.path.log_dir, study_name=study_name)
        study   = optuna.load_study(study_name=study_name, storage=storage)
        model   = src.engine.refit(X_train, y_train, study.best_trial, config.model_constants)

        # Predict the held-out test window and store the out-of-fold predictions.
        fold_preds = src.engine.predict(X_test, model, trf)  # Use the utility function to get predictions
        preds.append(fold_preds)

    # Combine all out-of-fold predictions and invert the target transform.
    actuals = df.set_index(['Store', 'Date'])['Sales']
    preds   = pd.concat(preds)
    results = pd.merge(left=actuals, right=preds,
                       left_index=True, right_index=True,
                       suffixes=('_actual', '_predicted')).sort_index()
    results.to_csv(config.path.predictions, index=True)
    logger.info("Stored out-of-fold predictions.")

    err_mae = mean_absolute_error(results["Sales_actual"], results["Sales_predicted"])
    logger.info(f"Out-of-fold MAE: {err_mae:.2f}")
