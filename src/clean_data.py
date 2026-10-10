from pathlib import Path

import pandas as pd

dfs = {p.stem: pd.read_parquet(p) for p in Path("data/raw").glob("*.parquet")}

# Combine all cities and drop the 77 empty hours at the start
df = pd.concat(dfs.values(), ignore_index=True)
df = df.dropna(subset=["pm2_5", "pm10"]).sort_values(["city", "time"]).reset_index(drop=True)

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/processed/all_cities.parquet", index=False)
print("Saved:", df.shape, "- rows, columns")
print("Rows per city:", df["city"].value_counts().unique())

# Check 1: Mumbai's five highest PM2.5 hours
print("\nMumbai top 5 PM2.5 hours:")
mumbai = df[df["city"] == "mumbai"]
print(mumbai.nlargest(5, "pm2_5")[["time", "pm2_5", "pm10"]])

# Check 2: longest run of identical PM2.5 values, per city
def longest_run(s):
    return s.ne(s.shift()).cumsum().value_counts().max()

print("\nLongest run of identical PM2.5 values (hours):")
print(df.groupby("city")["pm2_5"].apply(longest_run))