from pathlib import Path

import numpy as np
import pandas as pd

LAGS = [1, 3, 6, 12, 24, 48]
WINDOWS = [6, 24, 72]

FEATURES = (
    ["pm2_5", "pm10"]
    + [f"pm25_lag{k}" for k in LAGS]
    + [f"pm25_mean{w}h" for w in WINDOWS]
    + ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
       "wind_mean6h", "rain_sum24h"]
    + ["hour_sin", "hour_cos", "month_sin", "month_cos"]
)


def build_features(df):
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    def by_city(col):
        return df.groupby("city")[col]

    # PM2.5 from the past (shift(k) looks k hours back)
    for k in LAGS:
        df[f"pm25_lag{k}"] = by_city("pm2_5").shift(k)
    for w in WINDOWS:
        df[f"pm25_mean{w}h"] = by_city("pm2_5").transform(lambda s, w=w: s.rolling(w).mean())

    # Weather summaries
    df["wind_mean6h"] = by_city("wind_speed_10m").transform(lambda s: s.rolling(6).mean())
    df["rain_sum24h"] = by_city("precipitation").transform(lambda s: s.rolling(24).sum())

    # Time of day and month as circles
    hour = df["time"].dt.hour
    month = df["time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Target: PM2.5 exactly 24 hours later (shift(-24) looks forward)
    df["target"] = by_city("pm2_5").shift(-24)

    # The first and last rows of each city lack history or a target
    return df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_parquet("data/processed/all_cities.parquet")
    feats = build_features(raw)
    feats.to_parquet("data/processed/features.parquet", index=False)

    print("Saved:", feats.shape, "- rows, columns")
    print("Rows per city:", feats["city"].value_counts().unique())
    print("Number of features:", len(FEATURES))
    print("Any missing values left:", bool(feats[FEATURES + ["target"]].isna().any().any()))

    # Leakage check on Delhi: the target must be the value 24 rows later,
    # and lag24 must be the value 24 rows earlier
    d = feats[feats["city"] == "delhi"].reset_index(drop=True)
    print("Target check:", np.allclose(d["target"].iloc[:-24].values, d["pm2_5"].iloc[24:].values))
    print("Lag24 check: ", np.allclose(d["pm25_lag24"].iloc[24:].values, d["pm2_5"].iloc[:-24].values))
