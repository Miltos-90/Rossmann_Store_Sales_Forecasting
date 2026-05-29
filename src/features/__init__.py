"""
Feature engineering pipeline.

All features are constructed from the ``Sales`` column.
The ``lags``, ``diffs``, and ``roll_windows`` arguments passed to
``make_features`` are expressed as ``DateOffset`` objects or integer days.


Expected input columns for ``make_features``
---------------------------------------------
The DataFrame passed to ``make_features`` must contain the following columns:

=========================  ============================================================
Column                     Description
=========================  ============================================================
``Date``                   ``datetime64`` date of the observation.
``Store``                  Integer store identifier.
``Sales``                  Historical sales for the store on that date — the source
                           for all lag/rolling/diff features.
``DayOfWeek``              Day of week (1 = Monday … 7 = Sunday).
``Open``                   Binary flag — whether the store was open (1) or closed (0).
``Promo``                  Binary flag for the regular one-time promotion.
``Promo2``                 Binary flag indicating an active Promo2 interval.
``Promo2SinceDate``        Date from which Promo2 was active (``NaT`` if never).
``StateHoliday``           Public-holiday code (``'0'`` / ``0`` = none; ``'a'``/
                           ``'b'``/``'c'`` = holiday type).
``SchoolHoliday``          Binary flag for a school holiday.
``StoreType``              Store format category (``'a'``–``'d'``).
``Assortment``             Assortment level (``'a'``–``'c'``).
``CompetitionDistance``    Distance in metres to the nearest competitor store.
``CompetitionSinceDate``   Date the nearest competitor opened (used to compute
                           ``CompetitionSinceMonths``).
=========================  ============================================================

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
Named ``lag_<n>_<unit>`` where n/unit are derived from the DateOffset
(e.g. ``lag_7_days`` for a 7-day lag).
Each value is ``Sales`` looked up ``n <unit>`` before the current date.

Rolling-window statistics  (per window size × lag combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``lag_<lag>_roll_<w>_days_<stat>`` where ``<stat>`` ∈
{mean, std, skew, kurt, median, 10percentile, 90percentile}.
Computed over a window of ``w`` days on ``Sales`` starting from the lag offset.

First-order differences  (one pair per entry in ``diffs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``lag_1_<n>_<unit>_diff``        : absolute change in ``Sales`` over the period.
- ``lag_1_<n>_<unit>_pct_change``  : relative (%) change over the same period.

Target-encoded categoricals  (time-aware, per store)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Named ``<col>_te``.  For each categorical column listed below, the value is
the expanding historical mean of ``Sales`` for the (store, category) pair
up to — but **not including** — the current date, preventing lookahead leakage.

Encoded columns: ``Store``, ``Promo``, ``Promo2``, ``SchoolHoliday``,
``Assortment``, ``StoreType``, ``StateHoliday``, ``DayOfWeek``,
``Quarter``, ``Year``, ``Month``.


Sub-modules
-----------
- ``utils``           : shared helpers (to_list, pivot, melt, align)
- ``promo``           : promotion features (attach_store_data, make_consecutive_promo)
- ``holidays``        : holiday proximity features
- ``cyclic``          : cyclic sin/cos encoding
- ``lags``            : lag features and make_targets
- ``rolling``         : rolling-window statistics
- ``differences``     : first-order difference features
- ``target_encoding`` : time-aware per-(store, category) target encoding
- ``make_features``   : make_features (orchestrates all of the above)
"""

from .promo import attach_store_data
from .make_features import make_features
from .make_targets import make_targets

__all__ = ['make_features', 'make_targets', 'attach_store_data']
