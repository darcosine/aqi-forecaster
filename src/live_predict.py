import json
import time

import joblib
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from features import FEATURES
from fetch_data import CITIES
from live_features import make_features
from modeldata import FC, predict_range

# Retry timeouts, connection errors and server-side errors, pausing longer each time
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=4, connect=4, read=4, status=4, backoff_factor=5,
    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))


def get(url, params):
    r = SESSION.get(url, params=params, timeout=(20, 60))
    r.raise_for_status()
    return r


def forecast_now():
    """One 24-hour-ahead forecast per city, from the latest available hour."""
    models = joblib.load("models/models.joblib")
    offsets = {k: tuple(v) for k, v in json.load(open("models/offsets.json")).items()}
    columns = json.load(open("models/columns.json"))

    now = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None)
    rows = []
    for city, (lat, lon) in CITIES.items():
        common = dict(latitude=lat, longitude=lon, past_days=5, forecast_days=2,
                      timezone="Asia/Kolkata")
        aq = get("https://air-quality-api.open-meteo.com/v1/air-quality",
                 dict(common, hourly="pm2_5,pm10"))
        wx = get("https://api.open-meteo.com/v1/forecast",
                 dict(common, hourly="temperature_2m,relative_humidity_2m,"
                                     "wind_speed_10m,precipitation"))

        full = pd.DataFrame(aq.json()["hourly"]).merge(
            pd.DataFrame(wx.json()["hourly"]), on="time")
        full["time"] = pd.to_datetime(full["time"])
        full["city"] = city

        # Only the past may be used as history
        feats = make_features(full[full["time"] <= now].copy())
        last = feats.iloc[[-1]].copy()
        t = last["time"].iloc[0]
        if t != now:
            print(f"WARNING {city}: latest usable hour is {t}, not {now}")

        tgt = full[full["time"] == t + pd.Timedelta(hours=24)].iloc[0]
        last["fc_temp"] = tgt["temperature_2m"]
        last["fc_wind"] = tgt["wind_speed_10m"]
        last["fc_rain"] = tgt["precipitation"]
        last["fc_temp_change"] = last["fc_temp"] - last["temperature_2m"]

        X = last[FEATURES + FC].copy()
        X["city_" + city] = 1.0
        X = X.reindex(columns=columns, fill_value=0.0)

        rng = predict_range(models, X, last["pm2_5"].values, last["city"], offsets)
        rows.append({
            "made_at": t,
            "target_time": t + pd.Timedelta(hours=24),
            "city": city,
            "pm25_now": last["pm2_5"].iloc[0],
            "low": rng["q10"].iloc[0],
            "median": rng["q50"].iloc[0],
            "high": rng["q90"].iloc[0],
            "om_forecast": tgt["pm2_5"],
        })
        time.sleep(1)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = forecast_now()
    print("Forecast made at:", df["made_at"].iloc[0], "| for:", df["target_time"].iloc[0])
    print(df.drop(columns=["made_at", "target_time"]).round(1).to_string(index=False))