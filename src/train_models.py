import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from features import FEATURES

df = pd.read_parquet("data/processed/features.parquet")

# Add the city as one-hot columns (city_delhi, city_mumbai, ...)
cities = pd.get_dummies(df["city"], prefix="city", dtype=float)
X = pd.concat([df[FEATURES], cities], axis=1)
y = df["target"]

# Time-based split; drop the last 24 h of training so targets don't reach into the test period
SPLIT = pd.Timestamp("2025-07-01")
train = df["time"] < SPLIT - pd.Timedelta(hours=24)
test = df["time"] >= SPLIT
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]
print("Train rows:", len(X_train), "| Test rows:", len(X_test))

preds = {"baseline": X_test["pm2_5"].values}

lin = LinearRegression().fit(X_train, y_train)
preds["linear"] = lin.predict(X_test)

gbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                    random_state=42, verbose=-1)
gbm.fit(X_train, y_train)
preds["lightgbm"] = gbm.predict(X_test)

# Scores per city and overall
city_test = df.loc[test, "city"].values
mae, rmse = {}, {}
for name, p in preds.items():
    err = pd.Series(p - y_test.values)
    mae[name] = err.abs().groupby(city_test).mean()
    rmse[name] = (err**2).groupby(city_test).mean() ** 0.5

mae, rmse = pd.DataFrame(mae), pd.DataFrame(rmse)
print("\nMAE per city:")
print(mae.round(1))
print("\nRMSE per city:")
print(rmse.round(1))
print("\nAverage across cities:")
print(pd.DataFrame({"MAE": mae.mean(), "RMSE": rmse.mean()}).round(2))

# Most important features for LightGBM
imp = pd.Series(gbm.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 8 LightGBM features:")
print(imp.head(8))
