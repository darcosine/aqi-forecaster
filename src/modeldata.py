from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES

FC = ["fc_temp", "fc_wind", "fc_rain", "fc_temp_change"]


def load():
    """Features + day-before forecast weather, ready for modelling."""
    df = pd.read_parquet("data/processed/features.parquet")

    fc = (pd.concat([pd.read_parquet(p) for p in Path("data/forecast_prev").glob("*.parquet")],
                    ignore_index=True)
          .sort_values(["city", "time"]).reset_index(drop=True))
    g = fc.groupby("city")
    fc["fc_temp"] = g["temperature_2m_previous_day1"].shift(-24)
    fc["fc_wind"] = g["wind_speed_10m_previous_day1"].shift(-24)
    fc["fc_rain"] = g["precipitation_previous_day1"].shift(-24)

    df = df.merge(fc[["city", "time", "fc_temp", "fc_wind", "fc_rain"]],
                  on=["city", "time"], how="left")
    df["fc_temp_change"] = df["fc_temp"] - df["temperature_2m"]

    cities = pd.get_dummies(df["city"], prefix="city", dtype=float)
    X = pd.concat([df[FEATURES + FC], cities], axis=1)
    delta = df["target"] - df["pm2_5"]
    return df, X, delta


def predict_range(models, X, current, city, offsets):
    """Low / median / high PM2.5 estimates for 24 hours ahead, with per-city range shifts."""
    p = np.column_stack([current + models[k].predict(X) for k in ("q10", "q50", "q90")])
    p = np.clip(np.sort(p, axis=1), 0, None)
    city = pd.Series(np.asarray(city))
    lo = city.map(lambda c: offsets[c][0]).values
    hi = city.map(lambda c: offsets[c][1]).values
    q10 = np.minimum(np.clip(p[:, 0] - lo, 0, None), p[:, 1])
    q90 = np.maximum(p[:, 2] + hi, p[:, 1])
    return pd.DataFrame({"q10": q10, "q50": p[:, 1], "q90": q90})