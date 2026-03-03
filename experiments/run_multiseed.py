import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--width", type=int, required=True)
args = parser.parse_args()

seeds = [42, 123, 999]

for seed in seeds:
    for layer in range(12):

        subprocess.run([
            "python", "-m", "scripts.train_layer",
            "--config", args.config,
            "--layer", str(layer),
            "--seed", str(seed),
            "--width", str(args.width)
        ])

        subprocess.run([
            "python", "-m", "scripts.evaluate_layer",
            "--config", args.config,
            "--layer", str(layer),
            "--seed", str(seed),
            "--width", str(args.width)
        ])