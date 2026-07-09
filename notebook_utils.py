import pandas as pd
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(results, n_days_to_plot, n_days_offset):
        
    max_date = results.index.get_level_values("Date").max() - pd.Timedelta(days=n_days_offset)
    min_date = max_date - pd.Timedelta(days=n_days_to_plot)

    actuals_recent = results.loc[(slice(None), slice(min_date, max_date)), 'Sales_actual']
    preds_recent   = results.loc[(slice(None), slice(min_date, max_date)), 'Sales_predicted']

    plot_stores = actuals_recent.index.get_level_values("Store").unique()

    fig, ax = plt.subplots(figsize=(12, 2 * len(plot_stores)), nrows=len(plot_stores), sharex=True, sharey=False)

    for i, store in enumerate(plot_stores):

        actual_store_sales    = actuals_recent.loc[(store, slice(None))].copy()
        predicted_store_sales = preds_recent.loc[(store, slice(None))].copy()

        actual_store_sales.fillna(method='ffill', inplace=True)     # Forward fill to maintain continuity in the plot
        predicted_store_sales.fillna(method='ffill', inplace=True)  # Forward fill to maintain continuity in the plot

        actual_store_sales.plot(label="Actual", ax=ax[i])
        predicted_store_sales.plot(label="Predicted", ax=ax[i], linestyle='--')

        ax[i].set_ylabel("Store {}".format(store))
        ax[i].set_xlabel("Date")

    plt.suptitle("Actual vs Predicted Sales for Each Store (Last {} Days)".format(n_days_to_plot), fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
    plt.show()