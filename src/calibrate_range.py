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

# Three periods: fit the models, calibrate the range, test
H = pd.Timedelta(hours=24)
FIT_END = pd.Timestamp("2024-07-01")
SPLIT = pd.Timestamp("2025-07-01")
fit_rows = df["time"] < FIT_END - H
calib = (df["time"] >= FIT_END) & (df["time"] < SPLIT - H)
test = df["time"] >= SPLIT
print("Fit rows:", int(fit_rows.sum()), "| Calibration rows:", int(calib.sum()),
      "| Test rows:", int(test.sum()))

cities = pd.get_dummies(df["city"], prefix="city", dtype=float)
X = pd.concat([df[FEATURES + FC], cities], axis=1)
delta = df["target"] - df["pm2_5"]

models = {}
for a in (0.1, 0.5, 0.9):
    m = LGBMRegressor(objective="quantile", alpha=a, n_estimators=500, learning_rate=0.05,
                      num_leaves=31, random_state=42, verbose=-1)
    models[a] = m.fit(X[fit_rows], delta[fit_rows])


def predict(mask):
    cur = df.loc[mask, "pm2_5"].values
    p = np.column_stack([cur + models[a].predict(X[mask]) for a in (0.1, 0.5, 0.9)])
    p = np.clip(np.sort(p, axis=1), 0, None)
    out = df.loc[mask, ["city", "time", "target"]].copy()
    out["baseline"] = cur
    out["q10"], out["q50"], out["q90"] = p.T
    return out


cal, res = predict(calib), predict(test)

# Per-city corrections measured on the calibration year only
corr = {}
for city, d in cal.groupby("city"):
    corr[city] = (np.quantile(d["q10"] - d["target"], 0.9),
                  np.quantile(d["target"] - d["q90"], 0.9))
print("\nShift applied to each edge (negative lower shift = edge moves up):")
print(pd.DataFrame(corr, index=["lower edge", "upper edge"]).T.round(1))


def apply_corr(d):
    lo = d["city"].map(lambda c: corr[c][0])
    hi = d["city"].map(lambda c: corr[c][1])
    d["q10c"] = np.minimum(np.clip(d["q10"] - lo, 0, None), d["q50"])
    d["q90c"] = np.maximum(d["q90"] + hi, d["q50"])
    return d


cal, res = apply_corr(cal), apply_corr(res)


def coverage(d, lo, hi):
    t = d["target"]
    return pd.Series({
        "inside (%)": 100 * ((t >= d[lo]) & (t <= d[hi])).mean(),
        "above hi (%)": 100 * (t > d[hi]).mean(),
        "below lo (%)": 100 * (t < d[lo]).mean(),
        "mean width": (d[hi] - d[lo]).mean(),
    }).round(1)


thr = res.groupby("city")["target"].transform(lambda s: s.quantile(0.95))
winter = (res["time"] >= "2025-11-01") & (res["time"] < "2026-03-01")
slices = {"all hours": res, "spike hours": res[res["target"] >= thr], "winter": res[winter]}

rows = {("calibration year", "after"): coverage(cal, "q10c", "q90c")}
for name, d in slices.items():
    rows[(name, "before")] = coverage(d, "q10", "q90")
    rows[(name, "after")] = coverage(d, "q10c", "q90c")
print("\nCoverage (target: inside 80, above 10, below 10):")
print(pd.DataFrame(rows).T)

print("\nBy city, test period, after correction:")
by_city = pd.DataFrame({c: coverage(d, "q10c", "q90c") for c, d in res.groupby("city")}).T
by_city["MAE median"] = (res["q50"] - res["target"]).abs().groupby(res["city"]).mean().round(1)
print(by_city[["MAE median", "inside (%)", "above hi (%)", "mean width"]])

print("\nMAE averaged across cities - baseline:",
      round((res["baseline"] - res["target"]).abs().groupby(res["city"]).mean().mean(), 2),
      "| median model:",
      round((res["q50"] - res["target"]).abs().groupby(res["city"]).mean().mean(), 2))