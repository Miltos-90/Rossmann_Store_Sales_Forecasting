import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK

import src.constants as C

from src.cv import TimeSeriesCV
from src.callbacks import EarlyStoppingCV, LogEvalCallback

def make_objective(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: TimeSeriesCV,
    xgb_constants: dict,
    log_period: int = C.LOG_PERIOD,
):
    """
    Factory function that creates an objective function for hyperopt to minimize,
    which runs xgb.cv() with the given data, cross-validator, and fixed XGBoost parameters.
    The returned objective function takes a dictionary of XGBoost parameters to search over,
    runs cross-validation with early stopping, and returns the mean RMSPE of the test folds as the loss to minimize.

    Parameters
    ----------
    X : pd.DataFrame
        Outer-fold training data the inner CV will be run on.
    y : pd.DataFrame
        Outer-fold target data the inner CV will be run on.
    cv : TimeSeriesCV
        Pre-configured inner cross-validator.
    xgb_constants : dict
        Fixed XGBoost parameters that are not part of the search space.
    log_period : int
        Log CV metrics every ``log_period`` boosting rounds.

    Returns
    -------
    Callable[[dict], dict]
        Objective function that hyperopt can minimise (loss = mean RMSPE).
    """

    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    folds = list(cv.split(X))

    def objective(params: dict) -> dict:

        xgb_params = {**xgb_constants, **params}
        num_boost_round = xgb_params.pop("num_boost_round")
        early_stopping_rounds = xgb_params.pop("early_stopping_rounds")
        eval_metric = xgb_params["eval_metric"]

        # Fresh callbacks per trial to avoid stale state across hyperopt trials.
        cv_callbacks = [
            EarlyStoppingCV(rounds=early_stopping_rounds,
                            maximize=False,
                            metric_name=eval_metric,
                            data_name="test"),
            LogEvalCallback(eval_metric, period=log_period),
        ]

        cv_res = xgb.cv(params=xgb_params,
                        dtrain=dtrain,
                        num_boost_round=num_boost_round,
                        folds=folds,
                        metrics=eval_metric,
                        as_pandas=True,
                        maximize=False,
                        seed=0,
                        callbacks=cv_callbacks)

        loss = cv_res[f'test-{eval_metric}-mean'].iloc[-1]

        # If early stopping triggered, best_iteration is the optimal number of rounds; otherwise, use the max.
        params["num_boost_round"] = cv_res.attrs.get("best_iteration", len(cv_res))  
        
        return {"loss": loss, "status": STATUS_OK, "params": params}

    return objective
