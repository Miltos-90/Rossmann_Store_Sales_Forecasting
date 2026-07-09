""" Utility functions for Optuna hyperparameter tuning and artifact management """

import logging
import os
import pandas as pd
import xgboost as xgb

from optuna.trial import Trial
from optuna.storages.journal import (
    JournalFileOpenLock, JournalStorage, JournalFileBackend
)

logger = logging.getLogger(__name__)

def storage(log_dir: str, study_name: str) -> JournalStorage:
    """Set up a journal-based storage for Optuna.

    Args:
        log_dir: Directory to store the journal file.
        study_name: Name of the Optuna study.
    Returns:
        An instance of JournalStorage for use with Optuna.

    """
    file_path = f"{log_dir}/{study_name}.journal"

    logger.debug(f"Setting up journal storage for study '{study_name}' on {file_path}.")

    lock_obj  = JournalFileOpenLock(file_path)
    backend   = JournalFileBackend(file_path, lock_obj=lock_obj)
    storage   = JournalStorage(backend)

    return storage


def trial(
    study_name: str,
    trial: Trial,
    history: pd.DataFrame,
    boosters: list[xgb.Booster] | None,
    log_dir: str,
) -> None:
    """Persist CV history and fold boosters to disk, keyed by study/trial.

    Args:
        study_name: Name of the Optuna study for directory organization.
        trial: Optuna trial object to store artifact metadata.
        history: DataFrame containing cross-validation history from xgb.cv().
        boosters: List of trained XGBoost booster objects (one per fold), or None.
        log_dir: Root directory for artifact storage.
    """
    trial_dir = os.path.join(log_dir, study_name, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save full CV history
    history_path = os.path.join(trial_dir, "cv_history.csv")
    history.to_csv(history_path, index=True)
    trial.set_user_attr("artifact_dir", trial_dir)
    trial.set_user_attr("cv_history_path", history_path)

    # Save fold boosters
    if boosters:
        booster_paths = []
        for i, bst in enumerate(boosters):
            bst_path = os.path.join(trial_dir, f"fold_{i}.ubj")
            bst.save_model(bst_path)
            booster_paths.append(bst_path)
        trial.set_user_attr("booster_paths", booster_paths)

    logger.debug(f"Trial {trial.number} artifacts saved to {trial_dir}")
