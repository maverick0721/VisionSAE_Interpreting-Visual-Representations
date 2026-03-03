import os
import json
import pandas as pd

results_dir = "results/raw"

rows = []

for file in os.listdir(results_dir):
    if file.endswith(".json"):
        with open(os.path.join(results_dir, file)) as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows)
df.to_csv("results_summary.csv", index=False)

print("Saved results_summary.csv")