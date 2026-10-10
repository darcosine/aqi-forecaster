import pandas as pd
from lightgbm import LGBMRegressor

from features import FEATURES

df = pd.read_parquet("data/processed/features.parquet")
raw = (pd.read_parquet("data/processed/all_cities.parquet")
       .sort_values(["city", "time"]).reset_index(drop=True))

# "Oracle" columns: the real weather 24 hours ahead (look-ahead, only for this experiment)
g = raw.groupby("city")
oracle = raw[["city", "time"]].copy()
oracle["wx_temp"] = g["temperature_2m"].shift(-24)
oracle["wx_humid"] = g["relative_humidity_2m"].shift(-24)
oracle["wx_wind"] = g["wind_speed_10m"].shift(-24)
oracle["wx_rain"] = g["precipitation"].shift(-24)
oracle["wx_wind_mean24"] = g["wind_speed_10m"].transform(lambda s: s.rolling(24).mean().shift(-24))
oracle["wx_rain_sum24"] = g["precipitation"].transform(lambda s: s.rolling(24).sum().shift(-24))
ORACLE = [c for c in oracle.columns if c.startswith("wx_")]

df = df.merge(oracle, on=["city", "time"], how="left")
print("Missing oracle values:", int(df[ORACLE].isna().sum().sum()))

cities = pd.get_dummies(df["city"], prefix="city", dtype=float)
XA = pd.concat([df[FEATURES], cities], axis=1)
XB = pd.concat([df[FEATURES + ORACLE], cities], axis=1)
y = df["target"]

SPLIT = pd.Timestamp("2025-07-01")
train = df["time"] < SPLIT - pd.Timedelta(hours=24)
test = df["time"] >= SPLIT


def fit(X):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          random_state=42, verbose=-1)
    return model.fit(X[train], y[train])


res = df.loc[test, ["city", "time", "target"]].copy()
res["baseline"] = df.loc[test, "pm2_5"]
res["lgbm"] = fit(XA).predict(XA[test])
res["lgbm_oracle"] = fit(XB).predict(XB[test])
models = ["baseline", "lgbm", "lgbm_oracle"]


def summarize(d):
    rows = {}
    for m in models:
        e = d[m] - d["target"]
        g = e.groupby(d["city"])
        rows[m] = {
            "MAE": g.apply(lambda s: s.abs().mean()).mean(),
            "RMSE": g.apply(lambda s: (s ** 2).mean() ** 0.5).mean(),
            "bias": g.mean().mean(),
        }
    return pd.DataFrame(rows).T.round(2)


print("\nAll test hours (average across cities):")
print(summarize(res))

thr = res.groupby("city")["target"].transform(lambda s: s.quantile(0.95))
print("\nSpike hours (top 5% per city):")
print(summarize(res[res["target"] >= thr]))

winter = res[(res["time"] >= "2025-11-01") & (res["time"] < "2026-03-01")]
print("\nWinter (Nov 2025 - Feb 2026):")
print(summarize(winter))
