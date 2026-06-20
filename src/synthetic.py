import numpy as np
import pandas as pd

SEED     = 42

def sales_data(n_stores, days, seasonal_magnitude, base_magnitude, trend_magnitude, promo_magnitude, seed=SEED):
    dates    = pd.date_range("2013-01-01", periods=days, freq="D")
    rows = []
    rng  = np.random.default_rng(seed)
    for store_id in range(1, n_stores + 1):
        base = base_magnitude + store_id * base_magnitude * 0.25
        for date in dates:
            dow      = date.dayofweek + 1    # 1=Mon … 7=Sun
            is_sun   = dow == 7
            seasonal = seasonal_magnitude * np.sin(2 * np.pi * (dow - 1) / 6)   # peak Wed
            trend    = trend_magnitude * store_id * (date - dates[0]).days
            promo    = promo_magnitude * int(rng.random() < 0.25)
            state_hol = "0" if rng.random() >= 0.1 else "a"  # 10% chance of state holiday
            school_hol = 1 if rng.random() < 0.15 else 0  # 15% chance of school holiday
            sales    = max(0.0, base + seasonal + trend + promo) if not is_sun else 0.0
            rows.append({
                "Date":          date,
                "Store":         store_id,
                "DayOfWeek":     dow,
                "Open":          0 if is_sun else 1,
                "Promo":         promo,
                "StateHoliday":  state_hol,
                "SchoolHoliday": school_hol,
                "Sales":         sales
            })

    train = pd.DataFrame(rows)

    return train

def store_data(n_stores):

    # Generate synthetic store data with some variability across stores
    stores = []
    for store_id in range(1, n_stores + 1):
        store_type = ["a", "b", "c"][(store_id - 1) % 3]
        assortment = ["a", "a", "b"][(store_id - 1) % 3]
        competition_distance = 500.0 + (store_id - 1) * 350
        competition_since_date = pd.to_datetime("2012-01-01") + pd.DateOffset(months=store_id * 3)
        promo2 = 1 if store_id % 2 == 1 else 0
        if promo2 == 1:
            promo2_since_date = pd.to_datetime("2012-01-01") + pd.DateOffset(months=store_id * 6)
            promo_interval = "Jan,Apr,Jul,Oct"
        else:
            promo2_since_date = pd.NaT
            promo_interval = ""

        record = {
            "Store":               store_id,
            "StoreType":           store_type,
            "Assortment":          assortment,
            "CompetitionDistance": competition_distance,
            "CompetitionSinceDate": competition_since_date,
            "Promo2SinceDate":     promo2_since_date,
            "PromoInterval":       promo_interval,
            "Promo2":              promo2
        }

        stores.append(record)

    stores = pd.DataFrame(stores)

    return stores
