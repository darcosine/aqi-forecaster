import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from modeldata import load, predict_range

df, X, delta = load()

H = pd.Timedelta(hours=24)
FIT_END = pd.Timestamp("2024-07-01")
SPLIT = pd.Timestamp("2025-07-01")
fit_rows = df["time"] < FIT_END - H
calib = (df["time"] >= FIT_END) & (df["time"] < SPLIT - H)
ALPHAS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def train(rows):
    models = {}
    for name, a in ALPHAS.items():
        m = LGBMRegressor(objective="quantile", alpha=a, n_estimators=500, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)
        models[name] = m.fit(X[rows], delta[rows])
    return models


# 1. Measure the per-city range shifts, same recipe as the last step
temp_models = train(fit_rows)
zero = {c: (0.0, 0.0) for c in df["city"].unique()}
cal = df.loc[calib, ["city", "target"]].reset_index(drop=True)
rng = predict_range(temp_models, X[calib], df.loc[calib, "pm2_5"].values, cal["city"], zero)
offsets = {}
for city in sorted(cal["city"].unique()):
    m = (cal["city"] == city).values
    offsets[city] = (float(np.quantile(rng.loc[m, "q10"] - cal.loc[m, "target"], 0.9)),
                     float(np.quantile(cal.loc[m, "target"] - rng.loc[m, "q90"], 0.9)))
print("Range shifts (lower edge, upper edge):")
print(pd.DataFrame(offsets, index=["lower", "upper"]).T.round(1))

# 2. Final models: trained on ALL rows
final = train(df["time"].notna())

# 3. Save everything the dashboard will need
out = Path("models")
out.mkdir(exist_ok=True)
joblib.dump(final, out / "models.joblib")
(out / "offsets.json").write_text(json.dumps(offsets, indent=2))
(out / "columns.json").write_text(json.dumps(list(X.columns)))
print("\nSaved to models/:", sorted(p.name for p in out.iterdir()))

# 4. Sanity check: reload from disk, predict Delhi's last 7 days
reloaded = joblib.load(out / "models.joblib")
off = {k: tuple(v) for k, v in json.loads((out / "offsets.json").read_text()).items()}
last = df[(df["city"] == "delhi") & (df["time"] >= df["time"].max() - pd.Timedelta(days=7))]
Xl = X.loc[last.index]
a = predict_range(final, Xl, last["pm2_5"].values, last["city"], off)
b = predict_range(reloaded, Xl, last["pm2_5"].values, last["city"], off)
print("\nReloaded models give identical predictions:", bool(np.allclose(a.values, b.values)))

show = pd.concat([last[["time", "pm2_5", "target"]].reset_index(drop=True), a], axis=1)
show.columns = ["forecast made at", "PM2.5 now", "actual 24h later", "low", "median", "high"]
print("\nDelhi, one forecast every 24 hours (in-sample, just to see it runs):")
print(show.iloc[::24].round(1).to_string(index=False))