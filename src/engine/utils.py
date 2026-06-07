""" Utility functions for training and evaluation pipelines """

import logging
import os
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Literal

from optuna.study import Study
from optuna.trial import Trial

logger = logging.getLogger(__name__)

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


def overwrite_closed_sales(
        sales: pd.Series,
        mode: Literal['interpolate', 'zero'],
        index_name: str = "Date"
        ) -> pd.Series:
    """ 
    Re-interpolate sales for days when the store was closed (Sundays) with non-zero sales on adjacent days.

    Args
        sales: Series indexed by ['Date', 'Store'].
        mode: Method to handle closed days. 'interpolate' fills closed days with interpolated values based on adjacent days' sales, while 'zero' fills closed days with zeros.
        index_name: Name of the index level representing the date.
    
    Returns
        Series with same structure as input, but with 'Sales' re-interpolated for closed days.
    """
    is_closed = sales.index.get_level_values(index_name).day_name() == 'Sunday'
    sales.loc[is_closed] = np.nan

    if mode == 'interpolate':
        sales = (sales
                 .unstack('Store')
                 .interpolate(method='time')
                 .stack('Store'))

    elif mode == 'zero':
        sales.loc[is_closed] = 0

    return sales


def _wide_to_long_predictions(s: pd.DataFrame) -> pd.Series:
    """ Convert wide-format predictions with MultiIndex (Date, Store) and columns for each day ahead
        to long-format predictions with MultiIndex (Date, Store, Forecast Date) and a single value column.

        Args:
            s: DataFrame with MultiIndex (Date, Store) and columns for each day ahead

        Returns:
            Series with MultiIndex (Date, Store, Forecast Date) and values containing the predictions.
    """

    # Reset index to make Date and Store regular columns
    s_long = s.reset_index()

    # Melt from wide to long format
    s_long = s_long.melt(id_vars=["Date", "Store"],
                         var_name="lead_column",
                         value_name="value")

    # Extract days ahead from column name (e.g., "lead_1_days" -> 1)
    s_long['days_ahead'] = (s_long['lead_column'].astype(int) + 1)

    # Calculate forecast date by adding days to the base Date
    s_long['Forecast Date'] = s_long['Date'] + pd.to_timedelta(s_long['days_ahead'], unit='D')

    # Create the final long-format dataframe with MultiIndex (Date, Store, Forecast Date)
    s_long = (s_long
              .set_index(['Date', 'Store', 'Forecast Date'])
              .drop(['lead_column', 'days_ahead'], axis=1)
              ['value']  # Implicitly convert to Series
              .sort_index())

    return s_long


def retrieve_sales(s: pd.Series, index: pd.Index) -> pd.Series:
    """ 
    Retrieve actual sales values for the given index from the original sales series.

    Args:
        s (pd.Series): Original sales series with MultiIndex (Date, Store).
        index (pd.Index): Index of predictions with MultiIndex (Date, Store).

    Returns:
        pd.Series: Actual sales values aligned with the predictions index.
    """
    s_index = index.droplevel("Date").swaplevel("Store")  # Map from (Date, Store) to (Store, Date) to align with original sales series index
    s_out = s.loc[s_index]
    return s_out


def predict(booster: xgb.Booster, X_test: pd.DataFrame) -> pd.Series:
    """ 
    Generate predictions for the given test data using the provided XGBoost booster.

    Args:
        booster: Trained XGBoost booster.
        X_test: Test features.

    Returns:
        Series with predictions.
    """
    test_dmatrix  = xgb.DMatrix(X_test, enable_categorical=True)
    log_preds_raw = booster.predict(test_dmatrix)
    log_preds = pd.DataFrame(data=log_preds_raw, index=X_test.index)
    preds_long = _wide_to_long_predictions(log_preds)
    preds = overwrite_closed_sales(preds_long, mode='zero', index_name="Forecast Date")
    return preds

