import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--width", type=int, required=True)
args = parser.parse_args()

for layer in range(12):
    subprocess.run([
        "python", "-m", "scripts.extract_features",
        "--config", args.config,
        "--layer", str(layer)
    ])

    subprocess.run([
        "python", "-m", "scripts.train_layer",
        "--config", args.config,
        "--layer", str(layer)
    ])

    subprocess.run([
        "python", "-m", "scripts.evaluate_layer",
        "--config", args.config,
        "--layer", str(layer)
    ])