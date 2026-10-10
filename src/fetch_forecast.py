import time
from pathlib import Path

import pandas as pd
import requests

from fetch_data import CITIES, START, END

OUT = Path("data/forecast")
OUT.mkdir(parents=True, exist_ok=True)
VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]

for name, (lat, lon) in CITIES.items():
    r = requests.get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params=dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                    hourly=",".join(VARS), timezone="Asia/Kolkata"),
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = name
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name}: {len(df):,} rows, {int(df[VARS].isna().sum().sum())} missing values")
    time.sleep(2)

# How closely does the forecast weather match the real (archive) weather?
fc = pd.concat([pd.read_parquet(p) for p in OUT.glob("*.parquet")], ignore_index=True)
arch = pd.read_parquet("data/processed/all_cities.parquet")[["city", "time"] + VARS]
m = arch.merge(fc, on=["city", "time"], suffixes=("", "_fc"))
print("\nMatched rows:", len(m))

corr = pd.DataFrame({
    v: m.groupby("city").apply(lambda d, v=v: d[v].corr(d[v + "_fc"]))
    for v in VARS
}).round(3)
print("\nCorrelation: forecast weather vs real weather")
print(corr)