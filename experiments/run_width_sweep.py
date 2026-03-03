import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

widths = [4096, 8192, 16384, 32768]

for width in widths:
    for layer in range(12):

        subprocess.run([
            "python", "-m", "scripts.extract_features",
            "--config", args.config,
            "--layer", str(layer)
        ])

        subprocess.run([
            "python", "-m", "scripts.train_layer",
            "--config", args.config,
            "--layer", str(layer),
            "--seed", str(args.seed),
            "--width", str(width)
        ])

        subprocess.run([
            "python", "-m", "scripts.evaluate_layer",
            "--config", args.config,
            "--layer", str(layer),
            "--seed", str(args.seed),
            "--width", str(width)
        ])