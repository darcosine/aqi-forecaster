import time
from pathlib import Path

import pandas as pd
import requests

CITIES = {
    "delhi": (28.61, 77.21),
    "mumbai": (19.08, 72.88),
    "kolkata": (22.57, 88.36),
    "bengaluru": (12.97, 77.59),
    "hyderabad": (17.39, 78.48),
    "guntur": (16.31, 80.44),
    "vijayawada": (16.51, 80.65),
}

START, END = "2022-08-01", "2026-10-04"
OUT = Path("data/raw")


def fetch_city(name, lat, lon):
    # Pollution data (PM2.5 and PM10)
    aq = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                    hourly="pm2_5,pm10", timezone="Asia/Kolkata"),
        timeout=60,
    )
    aq.raise_for_status()

    # Weather data (temperature, humidity, wind, rain)
    wx = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                    hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                    timezone="Asia/Kolkata"),
        timeout=60,
    )
    wx.raise_for_status()

    # Turn both responses into tables and join them on the timestamp
    df_aq = pd.DataFrame(aq.json()["hourly"])
    df_wx = pd.DataFrame(wx.json()["hourly"])
    df = df_aq.merge(df_wx, on="time")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = name
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in CITIES.items():
        df = fetch_city(name, lat, lon)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows, {df['pm2_5'].isna().sum()} missing PM2.5")
        time.sleep(2)  # short pause so we don't hammer the free API