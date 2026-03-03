import subprocess

layers = range(12)
widths = [4096, 8192, 16384]
seeds = [42, 123, 999]

for width in widths:
    for layer in layers:
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):

                subprocess.run([
                    "python", "-m", "experiments.run_stability",
                    "--config", "configs/vit_base_cifar.yaml",
                    "--layer", str(layer),
                    "--width", str(width),
                    "--seed1", str(seeds[i]),
                    "--seed2", str(seeds[j])
                ])