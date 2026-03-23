import argparse
import os
import subprocess


def checkpoint_path(layer, width, seed):
    return f"checkpoints/layer_{layer}_width_{width}_seed_{seed}.pt"


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_base_cifar.yaml")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--seeds", default="42,123,999")
    parser.add_argument("--width-pairs", default="4096:8192,8192:16384")
    args = parser.parse_args()

    layers = parse_int_list(args.layers)
    seeds = parse_int_list(args.seeds)
    width_pairs = []
    for pair in args.width_pairs.split(","):
        left, right = pair.split(":")
        width_pairs.append((int(left), int(right)))

    launched = 0
    skipped = 0

    for layer in layers:
        for seed in seeds:
            for width_small, width_large in width_pairs:
                ckpt_small = checkpoint_path(layer, width_small, seed)
                ckpt_large = checkpoint_path(layer, width_large, seed)

                if not (os.path.exists(ckpt_small) and os.path.exists(ckpt_large)):
                    print(
                        "Skipping"
                        f" layer={layer} seed={seed}"
                        f" widths=({width_small},{width_large})"
                        " because checkpoints are missing"
                    )
                    skipped += 1
                    continue

                result = subprocess.run(
                    [
                        "python",
                        "-m",
                        "experiments.run_cross_width",
                        "--config",
                        args.config,
                        "--layer",
                        str(layer),
                        "--seed",
                        str(seed),
                        "--width_small",
                        str(width_small),
                        "--width_large",
                        str(width_large),
                    ],
                    check=False,
                )
                if result.returncode == 0:
                    launched += 1
                else:
                    skipped += 1

    print(f"Cross-width sweep complete: ran={launched}, skipped={skipped}")


if __name__ == "__main__":
    main()