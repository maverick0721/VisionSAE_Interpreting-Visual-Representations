import os
import json
import pandas as pd

rows = []

for file in os.listdir("results/cross_width"):
    if file.endswith(".json"):
        with open(os.path.join("results/cross_width", file)) as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows)
df.to_csv("cross_width_summary.csv", index=False)

print("Saved cross_width_summary.csv")