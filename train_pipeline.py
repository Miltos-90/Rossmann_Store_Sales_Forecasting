""" Training pipeline. """
import os
import logging
import optuna
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error
)

import src

logger = logging.getLogger(__name__)




def main(args):

    """ Main function to run the training pipeline. """

    config = src.AppSettings.from_yaml(args.config)

    # Set up logging
    os.makedirs(config.path.log_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(level=logging.INFO,
                        filename=config.path.logs,
                        format="%(levelname)s  %(message)s")
    logger = logging.getLogger(__name__)

    # Read and preprocess the data
    sales, stores = src.load_data(config.path)
    ################################################## TODO: Remove this hardcoding
    stores_to_use = [1, 2, 3]
    stores = stores[stores['Store'].isin(stores_to_use)]
    ##################################################

    df = src.preprocess_data(sales, stores)
    X, y, trf = src.generate_dataset(df, config)

    # Run nested cross-validation
    num_days = X.index.get_level_values("Date").nunique()
    cv_sizes = src.compute_cv_sizes(num_days, config.horizon.days, config.cross_validation)
    outer_cv = src.TimeSeriesCV(n_splits=config.cross_validation.n_outer_splits,
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

        src.optimize(study, X_train, y_train, cv_sizes, config)

        booster = src.refit(X_train, y_train, study, config)
        preds   = src.predict(X_test, booster, trf)

        predictions.append(preds)

    predictions = pd.concat(predictions)

    # Store results table.
    actuals = df.set_index(['Store', 'Date'])['Sales']
    results = pd.merge(left=actuals, right=predictions,
                       left_index=True, right_index=True,
                       suffixes=('_actual', '_predicted')
                       ).sort_index()

    results.to_csv(config.path.predictions, index=True)
    logger.info("Out-of-fold predictions saved to {}".format(config.path.predictions))

    # Compute errors
    act = results["Sales_actual"].values
    pred = results["Sales_predicted"].values
    err_mae  = mean_absolute_error(act, pred)
    err_rmse = root_mean_squared_error(act, pred)
    logger.info(f"Out-of-fold MAE: {err_mae:.2f}")
    logger.info(f"Out-of-fold RMSE: {err_rmse:.2f}")

    return

if __name__ == "__main__":

    args = src.parse_args()
    main(args)
