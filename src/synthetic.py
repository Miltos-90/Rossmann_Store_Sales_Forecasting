import numpy as np
import pandas as pd

DAYS     = 4000
N_STORES = 3
SEED     = 42

def sales_data(n_stores=N_STORES, days=DAYS, seed=SEED):
    dates    = pd.date_range("2013-01-01", periods=days, freq="D")
    rows = []
    rng  = np.random.default_rng(seed)
    for store_id in range(1, n_stores + 1):
        base = 300 + store_id * 150          # stores differ in overall level
        for date in dates:
            dow      = date.dayofweek + 1    # 1=Mon … 7=Sun
            is_sun   = dow == 7
            seasonal = base + 20 * np.sin(2 * np.pi * (dow - 1) / 6)   # peak Wed
            trend    = 0.15 * (date - dates[0]).days
            promo    = int(rng.random() < 0.25)
            sales    = max(0.0, seasonal + trend + promo * 100) if not is_sun else 0.0
            rows.append({
                "Date":          date,
                "Store":         store_id,
                "DayOfWeek":     dow,
                "Open":          0 if is_sun else 1,
                "Promo":         promo,
                "StateHoliday":  "0",
                "SchoolHoliday": 0,
                "Sales":         sales,
            })

    train = pd.DataFrame(rows)

    return train

def store_data(n_stores=N_STORES):

    stores = pd.DataFrame({
        "Store":               list(range(1, n_stores + 1)),
        "StoreType":           ["a", "b", "c"],
        "Assortment":          ["a", "a", "b"],
        "CompetitionDistance": [500.0, 1200.0, 800.0],
        "CompetitionSinceDate": pd.to_datetime(["2012-03-01", "2011-06-01", "2013-01-01"]),
        "Promo2SinceDate":     pd.to_datetime(["2012-06-01", pd.NaT,       "2013-03-01"]),
        "PromoInterval":       ["Jan,Apr,Jul,Oct", None,                   "Mar,Jun,Sep,Dec"],
    })

    return stores

