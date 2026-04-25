"""
Feature engineering for the Rossmann Store Sales forecasting pipeline.

All features are constructed from ``Shifted_Sales`` (i.e. ``Sales`` shifted
forward by one day per store) to prevent any leakage of current-day sales
into the feature set.  The ``lags``, ``diffs``, and ``roll_windows`` arguments
passed to ``make_features`` are expressed in terms of the **original Sales
axis**; the function internally reduces every offset by one day to compensate
for the shift.

Features produced by ``make_features``
---------------------------------------

Competition
~~~~~~~~~~~
- ``CompetitionDistance``       : log1p-transformed distance (metres) to the nearest competitor.
- ``CompetitionSinceMonths``    : number of months elapsed since the nearest competitor opened.

Calendar / seasonality
~~~~~~~~~~~~~~~~~~~~~~
- ``Year``                      : calendar year.
- ``Month``                     : calendar month (1–12).
- ``Quarter``                   : calendar quarter (1–4).
- ``DayOfWeek``                 : day of week (1 = Monday … 7 = Sunday).
- ``is_weekend``                : bool – True for Saturday and Sunday.
- ``is_month_start``            : bool – True for days 1–3 of the month.
- ``is_month_end``              : bool – True for days 28–31 of the month.
- ``DayOfMonth_sin / _cos``     : cyclic (sin/cos) encoding of the day of month (period 31).
- ``WeekOfYear_sin / _cos``     : cyclic (sin/cos) encoding of the ISO week number (period 52).

Promotion
~~~~~~~~~
- ``Promo``                     : binary flag for the regular one-time promotion.
- ``Promo2``                    : binary flag indicating an active running Promo2 interval
                                  for the store on that date.
- ``consecutive_promo_days``    : number of consecutive days the store has been in a
                                  Promo streak up to and including the current day.
- ``consecutive_promo2_days``   : same for Promo2.

School / state holidays
~~~~~~~~~~~~~~~~~~~~~~~
- ``SchoolHoliday``             : binary flag for a school holiday.
- ``StateHoliday``              : categorical code for public holidays (0 = none, a/b/c = type).
- ``days_to_next_state_holiday``: days until the next public holiday for the store's state.
- ``days_since_last_state_holiday``: days since the most recent public holiday.
- ``days_to_next_school_holiday``: days until the next school holiday.

Lag features  (one column per lag in ``lags``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``lag_<n>_<unit>`` where n/unit are derived from the adjusted DateOffset
(e.g. ``lag_6_days`` corresponds to a 7-day lag on original Sales).
Each value is ``Shifted_Sales`` looked up ``n <unit>`` before the current date.

Rolling-window statistics  (per window size × lag combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``lag_<lag>_roll_<w>_days_<stat>`` where ``<stat>`` ∈
{mean, std, skew, kurt, median, 10percentile, 90percentile}.
Computed over a window of ``w`` trading days on ``Shifted_Sales`` starting
from the adjusted lag offset.

First-order differences  (one pair per entry in ``diffs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``lag_1_<n>_<unit>_diff``        : absolute change in ``Shifted_Sales`` over the period.
- ``lag_1_<n>_<unit>_pct_change``  : relative (%) change over the same period.

Target-encoded categoricals  (time-aware, per store)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``<col>_te``.  For each categorical column listed below, the value is
the expanding historical mean of ``Shifted_Sales`` for the (store, category)
pair up to — but **not including** — the current date, preventing lookahead
leakage.

Encoded columns: ``Store``, ``Promo``, ``Promo2``, ``SchoolHoliday``,
``Assortment``, ``StoreType``, ``StateHoliday``, ``DayOfWeek``,
``Quarter``, ``Year``, ``Month``.

Sub-modules
-----------
- ``utils``           : shared helpers (_to_list, _pivot, _melt, _align)
- ``promo``           : promotion features (attach_store_data, _make_consecutive_promo)
- ``holidays``        : holiday proximity features
- ``cyclic``          : cyclic sin/cos encoding
- ``lags``            : lag features and make_targets
- ``rolling``         : rolling-window statistics
- ``differences``     : first-order difference features
- ``target_encoding`` : time-aware per-(store, category) target encoding
- ``make_features``   : make_features (orchestrates all of the above)
"""

from .promo import attach_store_data
from .lags import make_targets
from .make_features import make_features

__all__ = ['make_features', 'make_targets', 'attach_store_data']
