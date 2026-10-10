from pathlib import Path

import pandas as pd

dfs = {p.stem: pd.read_parquet(p) for p in Path("data/raw").glob("*.parquet")}

# 1. Missing values in every column (using Delhi as the example)
print("Missing values per column (Delhi):")
print(dfs["delhi"].isna().sum())

# 2. PM2.5 summary per city, ignoring the 77 empty hours
pm = pd.DataFrame({n: d.set_index("time")["pm2_5"] for n, d in dfs.items()}).dropna()
print("\nPM2.5 summary per city:")
print(pm.describe().round(1).T)

# 3. How similar are Guntur and Vijayawada?
print("\nCorrelation Guntur vs Vijayawada:", round(pm["guntur"].corr(pm["vijayawada"]), 3))
print("Share of hours with identical values:", round((pm["guntur"] == pm["vijayawada"]).mean(), 3))
