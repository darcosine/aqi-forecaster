import time
from pathlib import Path

import pandas as pd
import requests

from fetch_data import CITIES, START, END

OUT = Path("data/forecast_prev")
OUT.mkdir(parents=True, exist_ok=True)
VARS = ["temperature_2m_previous_day1",
        "wind_speed_10m_previous_day1",
        "precipitation_previous_day1"]

for name, (lat, lon) in CITIES.items():
    r = requests.get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        params=dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                    hourly=",".join(VARS), timezone="Asia/Kolkata"),
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = name
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name}: {len(df):,} rows")
    time.sleep(2)

# Where are the gaps?
fc = pd.concat([pd.read_parquet(p) for p in OUT.glob("*.parquet")], ignore_index=True)
is_test = fc["time"] >= pd.Timestamp("2025-07-01")

share = pd.DataFrame({
    "before 2025-07 (train)": fc.loc[~is_test, VARS].isna().mean(),
    "from 2025-07 (test)": fc.loc[is_test, VARS].isna().mean(),
}).T.round(3)
print("\nShare of missing values by period (all cities):")
print(share)

print("\nFirst hour with wind data, per city:")
print(fc.dropna(subset=[VARS[1]]).groupby("city")["time"].min())
