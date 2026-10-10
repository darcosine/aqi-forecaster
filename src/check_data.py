from pathlib import Path

import pandas as pd

dfs = {p.stem: pd.read_parquet(p) for p in Path("data/raw").glob("*.parquet")}

delhi = dfs["delhi"]
missing = delhi[delhi["pm2_5"].isna()]

print("First missing hour:", missing["time"].min())
print("Last missing hour: ", missing["time"].max())
print("Missing hours:     ", len(missing))

span = (missing["time"].max() - missing["time"].min()) / pd.Timedelta(hours=1) + 1
print("One continuous block:", span == len(missing))

gaps = {n: set(d.loc[d["pm2_5"].isna(), "time"]) for n, d in dfs.items()}
print("Same hours missing in every city:", all(g == gaps["delhi"] for g in gaps.values()))
