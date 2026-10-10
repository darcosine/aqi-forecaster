from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

from features import FEATURES

df = pd.read_parquet("data/processed/features.parquet")

# Day-before forecasts for the TARGET hour (t + 24h), same as the last step
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
y = df["target"]
delta = y - df["pm2_5"]  # the change over the next 24 hours


def fit(target):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          random_state=42, verbose=-1)
    return model.fit(X[train], target[train])


res = df.loc[test, ["city", "time", "target"]].copy()
res["baseline"] = df.loc[test, "pm2_5"]
res["lgbm_fc"] = fit(y).predict(X[test])
res["lgbm_delta"] = df.loc[test, "pm2_5"] + fit(delta).predict(X[test])
models = ["baseline", "lgbm_fc", "lgbm_delta"]


def summarize(d):
    rows = {}
    for m in models:
        e = d[m] - d["target"]
        gr = e.groupby(d["city"])
        rows[m] = {
            "MAE": gr.apply(lambda s: s.abs().mean()).mean(),
            "RMSE": gr.apply(lambda s: (s ** 2).mean() ** 0.5).mean(),
            "bias": gr.mean().mean(),
        }
    return pd.DataFrame(rows).T.round(2)


print("All test hours (average across cities):")
print(summarize(res))

thr = res.groupby("city")["target"].transform(lambda s: s.quantile(0.95))
print("\nSpike hours (top 5% per city):")
print(summarize(res[res["target"] >= thr]))

winter = res[(res["time"] >= "2025-11-01") & (res["time"] < "2026-03-01")]
print("\nWinter (Nov 2025 - Feb 2026):")
print(summarize(winter))

print("\nMAE per city (all test hours):")
print(pd.DataFrame({m: (res[m] - res["target"]).abs().groupby(res["city"]).mean()
                    for m in models}).round(1))