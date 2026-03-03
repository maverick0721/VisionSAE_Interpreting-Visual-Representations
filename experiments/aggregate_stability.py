import os
import json
import pandas as pd

rows = []

for file in os.listdir("results/stability"):
    if file.endswith(".json"):
        with open(os.path.join("results/stability", file)) as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows)
df.to_csv("stability_summary.csv", index=False)

print("Saved stability_summary.csv")