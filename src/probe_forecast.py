import pandas as pd
import requests

r = requests.get(
    "https://historical-forecast-api.open-meteo.com/v1/forecast",
    params=dict(latitude=28.61, longitude=77.21,
                start_date="2022-08-01", end_date="2026-10-04",
                hourly="temperature_2m,wind_speed_10m,precipitation",
                timezone="Asia/Kolkata"),
    timeout=60,
)
print("HTTP status:", r.status_code)

if r.ok:
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    print("Rows:", len(df))
    print("Missing per column:")
    print(df.isna().sum())
    ok = df.dropna()
    print("First hour with data:", ok["time"].min())
    print("Last hour with data: ", ok["time"].max())
else:
    print(r.text[:500])