""" Utility functions for training and evaluation pipelines """

import logging
import os
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import cm

from optuna.study import Study
from optuna.trial import Trial

logger = logging.getLogger(__name__)

def plot_results(actual: pd.DataFrame, predictions: pd.DataFrame):
    """
    Plot actual vs predicted sales for each store and each start date in a grid of line plots.
    
    Args
    actual : pd.DataFrame
        DataFrame containing the actual sales with a MultiIndex of (Date, Store).
    predictions : pd.DataFrame
        DataFrame containing the predicted sales with a MultiIndex of (Date, Store).

    Returns
    None
    """
    stores = actual.index.get_level_values("Store").unique().tolist()
    dates = actual.index.get_level_values("Date").unique().tolist()
    num_stores = len(stores)
    forecast_horizon = len(dates)

    fig, axes = plt.subplots(figsize=(num_stores * 3, forecast_horizon * 3),
                                nrows=forecast_horizon,
                                ncols=num_stores,
                                sharex=True,
                                sharey=True)
    colors = cm.Set1(np.linspace(0, 1, num_stores))   # one unique color per store

    for col, store in enumerate(stores):
        for row, start_date in enumerate(dates):

            # Get the actual and predicted sales for this store and this start date. 
            # Both are Series with index lead_1_days, lead_2_days, … lead_N_days.
            actual_store_date = actual.loc[(start_date, store)]
            predicted_store_date = predictions.loc[(start_date, store)]
            
            ax = axes[row, col]
            ax.plot(actual_store_date, marker='.', color=colors[store - 1], label=f"actual")
            ax.plot(predicted_store_date, linestyle='--', marker='x', color=colors[store - 1], label=f"predicted")

            # Set the title on the top row
            if ax == axes[0, col]:
                ax.set_title(f"Store {store}", fontsize=14, y=1.6)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2, fontsize=10)

            # Rotate x-axis labels on the last row
            if ax == axes[-1, col]:
                ticks = range(forecast_horizon)
                tick_labels = [f"{i+1}_days_ahead" for i in range(forecast_horizon)]
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)

            # Set y-axis label on the first column
            if col == 0:
                ax.set_ylabel(f"{start_date.date()}", fontsize=6)

    plt.tight_layout()

    return


def save_trial_artifacts(
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


def log_study(study: Study) -> None:
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

    if durations:
        summary["duration_total_s"] = round(sum(durations), 1)
        summary["duration_mean_s"]  = round(float(np.mean(durations)), 2)
        summary["duration_max_s"]   = round(float(max(durations)), 2)

    if complete_values:
        summary["best_value"]   = round(float(min(complete_values)), 6)
        summary["worst_value"]  = round(float(max(complete_values)), 6)
        summary["median_value"] = round(float(np.median(complete_values)), 6)

    best_trial = study.best_trial
    summary["best_trial_number"] = best_trial.number
    summary["best_trial_params"] = best_trial.params  # already a dict of JSON-serializable primitives

    study.set_user_attr("summary", summary)
    logger.info("Study '%s' summary persisted to Optuna DB.", study.study_name)

    return


def log_trial_cv_results(
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
    test_mean_col = f"test-{metric}-mean"
    test_std_col  = f"test-{metric}-std"
    final_loss      = history[test_mean_col].values[-1]
    final_loss_std  = history[test_std_col].values[-1] if test_std_col in history.columns else float("nan")
    best_loss       = history[test_mean_col].min()
    best_n_rounds = int(history[test_mean_col].idxmin()) + 1  # 1-based

    trial.set_user_attr("best_n_rounds",      best_n_rounds)
    trial.set_user_attr("final_loss_mean",    float(final_loss))
    trial.set_user_attr("final_loss_std",     float(final_loss_std))
    trial.set_user_attr("best_loss_mean",     float(best_loss))

    logger.debug(
        f"Trial {trial.number} | CV finished — "
        f"rounds used: {best_n_rounds}, "
        f"final {metric}: {final_loss:.4f} ± {final_loss_std:.4f}, "
        f"best {metric}: {best_loss:.4f} at round {best_n_rounds}"
    )

