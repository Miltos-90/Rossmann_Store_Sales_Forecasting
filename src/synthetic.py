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
            seasonal = 20 * np.sin(2 * np.pi * (dow - 1) / 6)   # peak Wed
            trend    = 0.5 * store_id * (date - dates[0]).days
            promo    = int(rng.random() < 0.25)
            sales    = max(0.0, base + seasonal + trend + promo * 100) if not is_sun else 0.0
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

    # Generate synthetic store data with some variability across stores
    stores = []
    for store_id in range(1, n_stores + 1):
        store_type = ["a", "b", "c"][(store_id - 1) % 3]
        assortment = ["a", "a", "b"][(store_id - 1) % 3]
        competition_distance = 500.0 + (store_id - 1) * 350
        competition_since_date = pd.to_datetime("2012-01-01") + pd.DateOffset(months=store_id * 3)
        promo2_since_date = pd.to_datetime("2012-01-01") + pd.DateOffset(months=store_id * 6) if store_id % 2 == 1 else pd.NaT
        promo_interval = "Jan,Apr,Jul,Oct" if store_id % 2 == 1 else None

        record = {
            "Store":               store_id,
            "StoreType":           store_type,
            "Assortment":          assortment,
            "CompetitionDistance": competition_distance,
            "CompetitionSinceDate": competition_since_date,
            "Promo2SinceDate":     promo2_since_date,
            "PromoInterval":       promo_interval,
        }

        stores.append(record)

    stores = pd.DataFrame(stores)

    return stores
