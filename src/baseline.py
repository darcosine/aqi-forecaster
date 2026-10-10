import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/all_cities.parquet")
df = df.sort_values(["city", "time"]).reset_index(drop=True)

# Target: the PM2.5 value exactly 24 hours later, within the same city
df["target"] = df.groupby("city")["pm2_5"].shift(-24)
df = df.dropna(subset=["target"])

# Time-based split: train on the past, test on the future
SPLIT = pd.Timestamp("2025-07-01")
train = df[df["time"] < SPLIT]
test = df[df["time"] >= SPLIT]
print("Train rows:", len(train), "| Test rows:", len(test))
print("Test period:", test["time"].min(), "to", test["time"].max())

# Baseline: predict that PM2.5 in 24 hours equals PM2.5 right now
err = test["pm2_5"] - test["target"]
res = (
    test.assign(abs_err=err.abs(), sq_err=err**2)
    .groupby("city")
    .agg(mean_pm25=("target", "mean"), MAE=("abs_err", "mean"), mse=("sq_err", "mean"))
)
res["RMSE"] = np.sqrt(res.pop("mse"))
print("\nBaseline (tomorrow = today) on the test period:")
print(res.round(1))
