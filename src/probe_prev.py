import pandas as pd
import requests

r = requests.get(
    "https://previous-runs-api.open-meteo.com/v1/forecast",
    params=dict(
        latitude=28.61, longitude=77.21,
        start_date="2022-08-01", end_date="2026-10-04",
        hourly="temperature_2m_previous_day1,wind_speed_10m_previous_day1,precipitation_previous_day1",
        timezone="Asia/Kolkata",
    ),
    timeout=60,
)
print("HTTP status:", r.status_code)

if not r.ok:
    print(r.text[:500])
else:
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    print("Rows:", len(df))
    print("Missing per column:")
    print(df.isna().sum())

    ok = df.dropna()
    print("First hour with data:", ok["time"].min())
    print("Last hour with data: ", ok["time"].max())

    # Compare the day-before forecast with the real (archive) weather for Delhi
    real = pd.read_parquet("data/raw/delhi.parquet")
    m = real.merge(df, on="time").dropna()
    for v in ["temperature_2m", "wind_speed_10m", "precipitation"]:
        c = m[v].corr(m[v + "_previous_day1"])
        mae = (m[v] - m[v + "_previous_day1"]).abs().mean()
        print(f"{v}: correlation {c:.3f}, mean abs difference {mae:.2f}")
