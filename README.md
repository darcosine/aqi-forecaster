\# AQI Forecaster: 24-hour PM2.5 forecasts for 7 Indian cities



Predicts PM2.5 24 hours ahead for Delhi, Mumbai, Kolkata, Bengaluru, Hyderabad, Guntur and Vijayawada, with a low/median/high range. A GitHub Action logs a new forecast every day, and a Streamlit dashboard shows the latest one.



\*\*Live dashboard:\*\* https://aqi-forecaster-8078.streamlit.app



\## What it does



\- Pulls hourly PM2.5, PM10 and weather for each city from \[Open-Meteo](https://open-meteo.com) (Aug 2022 to Oct 2026).

\- Builds features from recent PM2.5 history, current weather, day-before weather forecasts, and time of day and month.

\- Trains LightGBM quantile models that predict the 24-hour change in PM2.5 (10th, 50th and 90th percentile), then calibrates the range per city on a held-out year.

\- Every day at about 06:00 India time, a GitHub Action makes a fresh forecast and appends it to `logs/forecast\_log.csv`, next to Open-Meteo's own forecast, so the two can be compared on the same hours.



\## Results



Test period: July 2025 to October 2026 (about 15 months, including the whole 2025-26 winter). Models were trained only on earlier data; the split is by time, never random. Scores are averaged over the 7 cities.



| Model | MAE (µg/m³) | RMSE (µg/m³) |

|---|---|---|

| Baseline: tomorrow = today | 11.21 | 16.47 |

| Linear regression | 10.70 | 15.26 |

| LightGBM | 10.54 | 15.21 |

| LightGBM + day-before weather forecasts | 10.16 | 14.81 |



The median forecast of the final setup beats the baseline by roughly 9 to 10%. The range models were scored separately: the calibrated 10-90% range contained the true value 79.6% of the time on the test period (target: 80%).



\## What does not work well



\- \*\*Pollution spikes.\*\* On the worst 5% of hours in each city, the models do worse than the baseline (MAE about 31 vs 26) and under-predict by about 30 µg/m³. Weather forecasts did not fix this.

\- \*\*Winter.\*\* The calibrated range was still too narrow in winter: about 15% of winter hours fell above the high estimate (target 10%).

\- \*\*Gains are modest.\*\* "Tomorrow = today" is already a strong baseline for this data.

\- The Guntur and Vijayawada series are about 97% correlated, so they carry little independent information.



\## Data caveat



Open-Meteo's PM2.5 comes from an atmospheric model, not from ground sensors. Its values are smoother than real sensor readings, so these scores are probably more flattering than they would be against CPCB station data. Comparing against ground stations is the obvious next step.



\## Open question



Open-Meteo publishes its own PM2.5 forecast. The daily log stores it next to this project's forecast and the "tomorrow = today" baseline. With enough days logged, we can see which one is best. I do not know yet which will win.



\## Method notes



\- Weather inputs during training are the day-before forecasts (Open-Meteo previous-runs data, available from Jan 2024 and left blank before that). An "oracle" test with perfect weather showed the upper bound of what weather forecasts could add.

\- A feature-building check confirms the live pipeline produces identical features to the training pipeline.

\- Air quality bands in the dashboard follow CPCB categories, which are defined for 24-hour averages; the labels are approximate for a single-hour forecast.



\## Repo layout



\- `src/` data download, cleaning, features, training, evaluation, live forecast and dashboard scripts

\- `models/` saved final models and range calibration

\- `logs/forecast\_log.csv` the growing forecast log

\- `.github/workflows/daily\_forecast.yml` the daily job



\## Run it yourself



uv sync
uv run python src/fetch_data.py       # download hourly history
uv run python src/fetch_prev.py       # download day-before weather forecasts
uv run python src/clean_data.py
uv run python src/features.py
uv run python src/train_final.py      # train and save the models
uv run python src/log_forecast.py     # make and log a forecast
uv run streamlit run src/dashboard.py


Author: Spencer Antonio

