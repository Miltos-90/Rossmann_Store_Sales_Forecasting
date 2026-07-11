""" XGBoost callback for collecting fold boosters """

import xgboost as xgb
from typing import Optional

class BoosterCollector(xgb.callback.TrainingCallback):
    """Callback that captures the fold boosters from xgb.cv() after training ends."""

    def __init__(self) -> None:
        self.cvfolds: Optional[list[xgb.Booster]] = None

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        if hasattr(model, "cvfolds"):
            self.cvfolds = [fold.bst for fold in model.cvfolds]
        return model
