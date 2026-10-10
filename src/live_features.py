import numpy as np
import pandas as pd

from features import FEATURES, LAGS, WINDOWS


def make_features(df):
    """Same recipe as features.build_features, but keeps the newest rows,
    which have no 24-hour-ahead target yet."""
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    def by_city(col):
        return df.groupby("city")[col]

    for k in LAGS:
        df[f"pm25_lag{k}"] = by_city("pm2_5").shift(k)
    for w in WINDOWS:
        df[f"pm25_mean{w}h"] = by_city("pm2_5").transform(lambda s, w=w: s.rolling(w).mean())

    df["wind_mean6h"] = by_city("wind_speed_10m").transform(lambda s: s.rolling(6).mean())
    df["rain_sum24h"] = by_city("precipitation").transform(lambda s: s.rolling(24).sum())

    hour = df["time"].dt.hour
    month = df["time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df.dropna(subset=FEATURES).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_parquet("data/processed/all_cities.parquet")
    mine = make_features(raw)
    ref = pd.read_parquet("data/processed/features.parquet")

    m = ref[["city", "time"] + FEATURES].merge(
        mine[["city", "time"] + FEATURES], on=["city", "time"], suffixes=("_ref", "_new"))
    print("Rows compared:", len(m), "of", len(ref))

    diffs = {f: float((m[f + "_ref"] - m[f + "_new"]).abs().max()) for f in FEATURES}
    worst = max(diffs, key=diffs.get)
    print("Largest difference in any feature:", diffs[worst], f"({worst})")
    print("Identical:", all(v < 1e-9 for v in diffs.values()))

    print("\nNewest row in training features:", ref["time"].max())
    print("Newest row in live-style features:", mine["time"].max())