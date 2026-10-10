from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from features import FEATURES

df = pd.read_parquet("data/processed/features.parquet")

# Day-before forecasts for the TARGET hour (t + 24h), same as before
fc = (pd.concat([pd.read_parquet(p) for p in Path("data/forecast_prev").glob("*.parquet")],
                ignore_index=True)
      .sort_values(["city", "time"]).reset_index(drop=True))
g = fc.groupby("city")
fc["fc_temp"] = g["temperature_2m_previous_day1"].shift(-24)
fc["fc_wind"] = g["wind_speed_10m_previous_day1"].shift(-24)
fc["fc_rain"] = g["precipitation_previous_day1"].shift(-24)
FC = ["fc_temp", "fc_wind", "fc_rain", "fc_temp_change"]

df = df.merge(fc[["city", "time", "fc_temp", "fc_wind", "fc_rain"]],
              on=["city", "time"], how="left")
df["fc_temp_change"] = df["fc_temp"] - df["temperature_2m"]

SPLIT = pd.Timestamp("2025-07-01")
train = df["time"] < SPLIT - pd.Timedelta(hours=24)
test = df["time"] >= SPLIT

cities = pd.get_dummies(df["city"], prefix="city", dtype=float)
X = pd.concat([df[FEATURES + FC], cities], axis=1)
delta = df["target"] - df["pm2_5"]
cur = df.loc[test, "pm2_5"].values


def fit_quantile(alpha):
    model = LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=500,
                          learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    return model.fit(X[train], delta[train])


# Predict current value + predicted change; sort so low <= mid <= high; no negatives
preds = np.column_stack([cur + fit_quantile(a).predict(X[test]) for a in (0.1, 0.5, 0.9)])
preds = np.clip(np.sort(preds, axis=1), 0, None)

res = df.loc[test, ["city", "time", "target"]].copy()
res["baseline"] = cur
res["q10"], res["q50"], res["q90"] = preds.T


def summarize(d):
    mae_base = (d["baseline"] - d["target"]).abs().groupby(d["city"]).mean().mean()
    mae_mid = (d["q50"] - d["target"]).abs().groupby(d["city"]).mean().mean()
    return pd.Series({
        "MAE baseline": mae_base,
        "MAE median": mae_mid,
        "inside 10-90 (%)": 100 * ((d["target"] >= d["q10"]) & (d["target"] <= d["q90"])).mean(),
        "above q90 (%)": 100 * (d["target"] > d["q90"]).mean(),
        "below q10 (%)": 100 * (d["target"] < d["q10"]).mean(),
        "mean width": (d["q90"] - d["q10"]).mean(),
    }).round(1)


thr = res.groupby("city")["target"].transform(lambda s: s.quantile(0.95))
winter = (res["time"] >= "2025-11-01") & (res["time"] < "2026-03-01")

print(pd.DataFrame({
    "all hours": summarize(res),
    "spike hours": summarize(res[res["target"] >= thr]),
    "winter": summarize(res[winter]),
}).T)

print("\nBy city:")
print(pd.DataFrame({c: summarize(d) for c, d in res.groupby("city")}).T[
    ["MAE median", "inside 10-90 (%)", "above q90 (%)", "mean width"]])