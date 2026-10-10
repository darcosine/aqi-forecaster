import pandas as pd

df = pd.read_parquet("data/processed/all_cities.parquet")

# 1. Average PM2.5 by hour of day (every 3rd hour, to keep it short)
hourly = (
    df.assign(hour=df["time"].dt.hour)
    .groupby(["hour", "city"])["pm2_5"].mean()
    .unstack().round(0)
)
print("Average PM2.5 by hour of day:")
print(hourly.iloc[::3])

# 2. Correlation of PM2.5 with each weather variable
weather = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
corr = df.groupby("city")[weather + ["pm2_5"]].corr()["pm2_5"].unstack()[weather]
print("\nCorrelation of PM2.5 with weather:")
print(corr.round(2))

# 3. Correlation of PM2.5 with its own past values
lags = pd.DataFrame({
    f"{k}h_ago": df.groupby("city")["pm2_5"].apply(lambda s, k=k: s.corr(s.shift(k)))
    for k in (1, 24, 48)
}).round(2)
print("\nCorrelation of PM2.5 with its own past:")
print(lags)