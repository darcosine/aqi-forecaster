from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

LOG = Path("logs/forecast_log.csv")
BANDS = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"),
         (120, "Poor"), (250, "Very poor"), (float("inf"), "Severe")]


def category(value):
    for limit, name in BANDS:
        if value <= limit:
            return name


st.set_page_config(page_title="PM2.5 forecast", layout="wide")
st.title("PM2.5 forecast for the next 24 hours")

if not LOG.exists():
    st.error("No forecast log found yet.")
    st.stop()

log = pd.read_csv(LOG, parse_dates=["made_at", "target_time"])
latest = (log.sort_values("made_at").groupby("city").tail(1)
          .sort_values("median", ascending=False).reset_index(drop=True))

st.caption(f"Forecast made at {latest['made_at'].max():%d %b %Y, %H:%M} India time, "
           f"for {latest['target_time'].max():%d %b, %H:%M}. "
           f"Log has {len(log)} forecasts so far.")

cols = st.columns(4)
for i, r in latest.iterrows():
    with cols[i % 4]:
        st.metric(r["city"].title(), f"{r['median']:.0f} µg/m³",
                  f"now {r['pm25_now']:.0f}", delta_color="off")
        st.write(f"**{category(r['median'])}**")
        st.caption(f"Range {r['low']:.0f} to {r['high']:.0f}")

# Range chart: bar = low to high, black tick = our median, red diamond = Open-Meteo
fig, ax = plt.subplots(figsize=(10, 4))
y = range(len(latest))
ax.barh(list(y), latest["high"] - latest["low"], left=latest["low"],
        color="#9ecae1", height=0.5, label="Our range (low to high)")
ax.scatter(latest["median"], list(y), color="black", marker="|", s=300, zorder=3,
           label="Our median")
ax.scatter(latest["om_forecast"], list(y), color="red", marker="D", s=40, zorder=3,
           label="Open-Meteo forecast")
ax.set_yticks(list(y))
ax.set_yticklabels(latest["city"].str.title())
ax.invert_yaxis()
ax.set_xlabel("PM2.5 (µg/m³)")
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right")
st.pyplot(fig)

with st.expander("How to read this"):
    st.markdown(
        "- The **median** is our best guess. The **range** is meant to contain the real value "
        "about 80% of the time.\n"
        "- On past test data the median beat 'tomorrow = today' by roughly 4 to 10%, "
        "depending on the setup.\n"
        "- **Winter and pollution spikes are the weak spot.** The range was too narrow in winter, "
        "and the worst hours were under-predicted.\n"
        "- The data is modelled by Open-Meteo, not measured by ground sensors.\n"
        "- This forecast and Open-Meteo's own are logged daily so they can be compared fairly "
        "once enough days have passed."
    )

with st.expander("Raw log"):
    st.dataframe(log.sort_values("made_at", ascending=False), use_container_width=True)