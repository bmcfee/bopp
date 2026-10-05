#!/usr/bin/env python
# render_md.py

import json
import pandas as pd

with open("benchmark.json") as f:
    data = json.load(f)

df = pd.json_normalize(data["benchmarks"])

# pytest-benchmark outputs time metrics in seconds; scale to μs
time_cols = ["stats.min", "stats.mean", "stats.stddev"]
df[time_cols] *= 1e6

cols = {
    "params.loader_name": "Loader", 
    "stats.min": "Min (μs)", 
    "stats.mean": "Mean (μs)", 
    "stats.stddev": "StdDev (μs)", 
    "stats.iterations": "Rounds"
}

# Group by the parameterized prefix and render separate tables
for prefix, group in df.groupby("params.prefix"):
    print(f"### Prefix: {prefix}\n")
    
    group_df = group[cols.keys()].rename(columns=cols)
    # floatfmt ensures consistent decimal precision via tabulate
    print(group_df.to_markdown(index=False, floatfmt=".1f"))
    print("\n")
