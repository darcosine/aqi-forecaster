import pandas as pd
import requests

lat, lon = 28.61, 77.21  # Delhi

aq = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params=dict(latitude=lat, longitude=lon, hourly="pm2_5,pm10",
                past_days=4, forecast_days=2, timezone="Asia/Kolkata"),
    timeout=60,
)
wx = requests.get(
    "https://api.open-meteo.com/v1/forecast",
    params=dict(latitude=lat, longitude=lon,
                hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                past_days=4, forecast_days=2, timezone="Asia/Kolkata"),
    timeout=60,
)
print("HTTP status - air quality:", aq.status_code, "| weather:", wx.status_code)

now = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None)
print("Current hour (India time):", now)

for label, r in (("AIR QUALITY", aq), ("WEATHER", wx)):
    if not r.ok:
        print(label, "error:", r.text[:300])
        continue
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    ok = df.dropna()
    print(f"\n{label}: {len(df)} rows from {df['time'].min()} to {df['time'].max()}")
    print("  Missing per column:", df.isna().sum().drop("time").to_dict())
    print("  Last hour with complete data:", ok["time"].max())
    past = df[df["time"] <= now]
    print("  Missing in the past part (up to now):", int(past.isna().sum().sum()))
    print("  Values at the current hour:")
    print(df[df["time"] == now].drop(columns="time").round(1).to_string(index=False))