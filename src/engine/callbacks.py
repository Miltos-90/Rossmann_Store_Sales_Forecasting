import xgboost as xgb
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BoosterCollector(xgb.callback.TrainingCallback):
    """Callback that captures the fold boosters from xgb.cv() after training ends."""

    def __init__(self) -> None:
        self.cvfolds: Optional[list[xgb.Booster]] = None

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        if hasattr(model, "cvfolds"):
            self.cvfolds = [fold.bst for fold in model.cvfolds]
        return model

class EarlyStoppingCV(xgb.callback.EarlyStopping):
    """EarlyStopping that propagates best_iteration to each fold booster in xgb.cv().

    In some XGBoost versions CVPack.set_attr does not forward attributes to the
    underlying fold boosters, so xgb.cv() raises TypeError when reading
    cvfolds[0].bst.attr("best_iteration"). This subclass fixes that by
    explicitly mirroring the attribute on every fold booster after each update.
    """

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        """ Override of xgb.callback.EarlyStopping.after_iteration that also sets
            best_iteration on fold boosters. 

            Parameters
            ----------
            model : xgb.Booster
                The CVPack booster passed by xgb.cv() to the callback.
            epoch : int
                The current boosting round (0-based).
            evals_log : dict
                Evaluation results log passed by xgb.cv() to the callback.

            Returns
            -------
            bool
                True if training should stop, False otherwise.
        """
        should_stop = super().after_iteration(model, epoch, evals_log)
        if hasattr(model, "cvfolds"):
            best = model.attr("best_iteration")
            if best is not None:
                for fold in model.cvfolds:
                    fold.bst.set_attr(best_iteration=best)
        
        logger.debug(f"Iter {epoch+1}: best_iteration={model.attr('best_iteration')}, early_stop={should_stop}")
        return should_stop

class LogEvalCallback(xgb.callback.TrainingCallback):
    """Log evaluation metrics from xgb.cv() at a specified period."""

    def __init__(self, metric: str, period: int = 1):
        """ Initialize the callback. 
        
            Parameters
            ----------
            metric : str
                The name of the evaluation metric to log (e.g., "rmse").
            period : int
                The logging period (in boosting rounds).
        """
        self.metric = metric
        self.period = period

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        """ Override of xgb.callback.TrainingCallback.after_iteration that logs the specified metric from evals_log.

            Parameters
            ----------
            model : xgb.Booster
                The CVPack booster passed by xgb.cv() to the callback.
            epoch : int
                The current boosting round (0-based).
            evals_log : dict
                Evaluation results log passed by xgb.cv() to the callback.  

            Returns
            -------
            bool
                False (training should not stop based on this callback).
        """
        # CV stores metrics as tuples: (mean, std)
        if "test" in evals_log and self.metric in evals_log["test"]:
            mean, std = evals_log["test"][self.metric][-1]
            if (epoch + 1) % self.period == 0:
                logger.info(f"Iter {epoch+1}: test-{self.metric}-mean={mean:.6f}, std={std:.6f}")
        return False
