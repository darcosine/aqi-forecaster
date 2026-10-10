from pathlib import Path

import pandas as pd

from live_predict import forecast_now

LOG = Path("logs/forecast_log.csv")
LOG.parent.mkdir(exist_ok=True)

new = forecast_now()

if LOG.exists():
    old = pd.read_csv(LOG, parse_dates=["made_at", "target_time"])
    # Running twice in the same hour must not create duplicate rows
    log = pd.concat([old, new], ignore_index=True).drop_duplicates(
        ["made_at", "city"], keep="first")
    added = len(log) - len(old)
else:
    log, added = new, len(new)

log = log.sort_values(["made_at", "city"]).reset_index(drop=True)
log.to_csv(LOG, index=False)
print(f"Rows in log: {len(log)} ({added} new)")