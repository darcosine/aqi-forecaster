import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from features import FEATURES

df = pd.read_parquet("data/processed/features.parquet")
cities = pd.get_dummies(df["city"], prefix="city", dtype=float)
X = pd.concat([df[FEATURES], cities], axis=1)
y = df["target"]

SPLIT = pd.Timestamp("2025-07-01")
train = df["time"] < SPLIT - pd.Timedelta(hours=24)
test = df["time"] >= SPLIT

lin = LinearRegression().fit(X[train], y[train])
gbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                    random_state=42, verbose=-1).fit(X[train], y[train])

# Collect the test predictions in one table and save them
res = df.loc[test, ["city", "time", "target"]].copy()
res["baseline"] = X.loc[test, "pm2_5"]
res["linear"] = lin.predict(X[test])
res["lightgbm"] = gbm.predict(X[test])
res.to_parquet("data/processed/test_predictions.parquet", index=False)

models = ["baseline", "linear", "lightgbm"]
for m in models:
    res[f"err_{m}"] = (res[m] - res["target"]).abs()
errcols = [f"err_{m}" for m in models]

# 1. MAE by month of the forecast time (averaged across the 7 cities)
res["month"] = res["time"].dt.to_period("M")
by_month = res.groupby(["month", "city"])[errcols].mean().groupby("month").mean()
by_month.columns = models
by_month["lgbm_gain_%"] = (1 - by_month["lightgbm"] / by_month["baseline"]) * 100
print("MAE by month (average across cities):")
print(by_month.round(1))

# 2. The worst 5% of pollution hours in each city
thr = res.groupby("city")["target"].transform(lambda s: s.quantile(0.95))
spike = res[res["target"] >= thr]
print("\nSpike hours (top 5% per city) - MAE averaged across cities:")
print(spike.groupby("city")[errcols].mean().mean().round(1))
signed = (spike["lightgbm"] - spike["target"]).mean()
print("Average signed error of LightGBM on spikes:", round(signed, 1),
      "(negative = under-predicts)")
