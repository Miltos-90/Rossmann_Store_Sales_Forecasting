""" Training pipeline. """
import argparse
import os
import logging
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error
)

from utils import load_data, generate_dataset, refit, setup_logging
from engine import TimeSeriesCV, compute_cv_sizes, optimize, predict
from preprocessing import preprocess_data
from settings import AppSettings


setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--config", 
                        type=str, 
                        default="./config.yaml", 
                        help="Path to the configuration YAML file.")

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Path to the local input directory.")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Path to the local output directory.")

    return parser.parse_args()


def main(args):

    """ Main function to run the training pipeline. """
    config = AppSettings.from_yaml(args.config, args.input_dir, args.output_dir)
    os.makedirs(config.path.output_dir, exist_ok=True)

    # Read and preprocess the data
    sales, stores = load_data(config.path)
    df = preprocess_data(sales, stores)
    X, y, trf = generate_dataset(df, config)

    # Run nested cross-validation
    num_days = X.index.get_level_values("Date").nunique()
    cv_sizes = compute_cv_sizes(num_days, config.horizon.days, config.cross_validation)
    outer_cv = TimeSeriesCV(n_splits=config.cross_validation.n_outer_splits,
                            gap=config.horizon.days,
                            train_size=cv_sizes['outer_train'],
                            test_size=cv_sizes['outer_test'])

    predictions = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        logger.info(f"Outer fold {fold}/{config.cross_validation.n_outer_splits}")

        study   = f"outer_fold_{fold}"
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        optimize(study, X_train, y_train, cv_sizes, config)

        booster = refit(X_train, y_train, study, config)  # also saved in the output dir.
        preds   = predict(X_test, booster, trf)

        predictions.append(preds)

    predictions = pd.concat(predictions)

    # Store results table.
    actuals = df.set_index(['Store', 'Date'])['Sales']
    results = pd.merge(left=actuals, right=predictions,
                       left_index=True, right_index=True,
                       suffixes=('_actual', '_predicted')
                       ).sort_index()
    is_closed = results["Sales_actual"] == 0
    results.loc[is_closed, "Sales_predicted"] = 0  # Set predictions to 0 for closed stores (actual sales = 0)

    results.to_csv(config.path.predictions, index=True)
    logger.info(f"Out-of-fold predictions saved to {config.path.predictions}")

    # Compute errors
    act      = results["Sales_actual"].values
    pred     = results["Sales_predicted"].values
    err_mae  = mean_absolute_error(act, pred)
    err_rmse = root_mean_squared_error(act, pred)

    logger.info(f"Out-of-fold MAE: {err_mae:.2f}")
    logger.info(f"Out-of-fold RMSE: {err_rmse:.2f}")


if __name__ == "__main__":

    args = parse_args()
    main(args)
