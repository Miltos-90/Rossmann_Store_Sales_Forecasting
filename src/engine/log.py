""" Logging utilities for Optuna trials and studies """

import logging
import optuna
import numpy as np
import pandas as pd

from optuna.trial import Trial
from optuna.study import Study

logger = logging.getLogger(__name__)


def trial(
    trial: Trial,
    metric: str,
    history: pd.DataFrame,
) -> None:
    """Log CV results and store them as trial user attributes.

    Extracts best and final round metrics from CV history and stores them
    as trial user attributes for later analysis and reporting.

    Args:
        trial: Optuna trial object.
        metric: Metric name (e.g., "mae") used in XGBoost evaluation.
        history: DataFrame containing CV results for each boosting round,
            with columns like 'test-{metric}-mean' and 'test-{metric}-std'.
    """
    test_mean_col  = f"test-{metric}-mean"
    test_std_col   = f"test-{metric}-std"
    final_loss     = history[test_mean_col].values[-1]
    final_loss_std = history[test_std_col].values[-1] if test_std_col in history.columns else float("nan")
    best_loss      = history[test_mean_col].min()
    best_n_rounds  = int(history[test_mean_col].idxmin()) + 1  # 1-based

    trial.set_user_attr("best_n_rounds",   best_n_rounds)
    trial.set_user_attr("final_loss_mean", float(final_loss))
    trial.set_user_attr("final_loss_std",  float(final_loss_std))
    trial.set_user_attr("best_loss_mean",  float(best_loss))

    logger.info(
        f"Trial {trial.number} finished | "
        f"rounds used: {best_n_rounds}, "
        f"final {metric}: {final_loss:.4f} +- {final_loss_std:.4f}, "
        f"best {metric}: {best_loss:.4f} at round {best_n_rounds}"
    )


def study(study: Study) -> None:
    """Persist summary statistics for a completed Optuna study to the storage DB.

    Stores trial counts, duration statistics, objective value statistics, and
    the best trial's number and parameters as a single ``"summary"`` dict on
    the study via ``study.set_user_attr``.  Best-trial params and per-trial
    user attributes are already persisted individually by Optuna; this adds a
    study-level roll-up that can be queried without loading every trial.

    Args:
        study: A completed (or partial) Optuna study.
    """
    trials = study.trials

    # Compute summary statistics
    n_total   = len(trials)
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    n_pruned   = sum(t.state == optuna.trial.TrialState.PRUNED   for t in trials)
    n_failed   = sum(t.state == optuna.trial.TrialState.FAIL     for t in trials)
    p_pruned   = 100 * n_pruned / n_total if n_total > 0 else 0.0

    durations       = [t.duration.total_seconds() for t in trials if t.duration is not None]
    complete_values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

    summary: dict = {
        "n_total":    n_total,
        "n_complete": n_complete,
        "n_pruned":   n_pruned,
        "pct_pruned": round(p_pruned, 1),
        "n_failed":   n_failed,
    }
    
    # Compute duration and value statistics
    summary["duration_total_s"] = round(sum(durations), 1)
    summary["duration_mean_s"]  = round(float(np.mean(durations)), 2)
    summary["duration_max_s"]   = round(float(max(durations)), 2)
  
    summary["best_value"]   = round(float(min(complete_values)), 6)
    summary["worst_value"]  = round(float(max(complete_values)), 6)
    summary["median_value"] = round(float(np.median(complete_values)), 6)

    best_trial = study.best_trial
    summary["best_trial_number"] = best_trial.number
    summary["best_trial_params"] = best_trial.params

    # Store the summary as a study-level user attribute
    study.set_user_attr("summary", summary)
    logger.info(f"Study '{study.study_name}' summary persisted to optuna storage.")

    return
