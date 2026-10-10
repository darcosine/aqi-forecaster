import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet("data/processed/all_cities.parquet")

# Average PM2.5 for each calendar month (1 = January) and city
clim = (
    df.assign(month=df["time"].dt.month)
    .groupby(["month", "city"])["pm2_5"]
    .mean()
    .unstack()
    .round(0)
)
print(clim)

# Line chart of the monthly average over the whole timeline
monthly = df.set_index("time").groupby("city")["pm2_5"].resample("MS").mean().unstack(0)
fig, ax = plt.subplots(figsize=(12, 6))
monthly.plot(ax=ax)
ax.set_ylabel("Monthly mean PM2.5")
ax.set_title("Monthly average PM2.5 by city")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("notebooks/monthly_pm25.png", dpi=150)