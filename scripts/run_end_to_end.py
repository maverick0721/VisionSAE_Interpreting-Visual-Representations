import argparse
import itertools
import os
import subprocess
import sys
import tempfile

import torch
import yaml


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def run_cmd(cmd, allow_fail=False):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def make_runtime_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    requested = cfg["training"]["device"]
    using_temp = False

    if requested == "cuda" and not torch.cuda.is_available():
        cfg["training"]["device"] = "cpu"
        fd, tmp_path = tempfile.mkstemp(prefix="visionsae_runtime_", suffix=".yaml")
        os.close(fd)
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print("CUDA not available; using temporary CPU config for this run.")
        return tmp_path, True

    return config_path, using_temp


def adjacent_pairs(values):
    return list(zip(values[:-1], values[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_base_cifar.yaml")
    parser.add_argument("--layers", default="0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--widths", default="4096,8192")
    parser.add_argument("--skip-smoke-check", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick = lightweight demo; full = all layers/seeds/widths",
    )
    args = parser.parse_args()

    if args.mode == "full":
        layers = list(range(12))
        seeds = [42, 123, 999]
        widths = [4096, 8192, 16384]
    else:
        layers = parse_int_list(args.layers)
        seeds = parse_int_list(args.seeds)
        widths = parse_int_list(args.widths)

    if not layers or not seeds or not widths:
        raise ValueError("Layers, seeds, and widths must each contain at least one value")

    runtime_config, using_temp = make_runtime_config(args.config)

    print("\n=== VisionSAE End-to-End Run ===")
    print(f"Python: {sys.executable}")
    print(f"Config: {runtime_config}")
    print(f"Mode: {args.mode}")
    print(f"Layers: {layers}")
    print(f"Seeds: {seeds}")
    print(f"Widths: {widths}")

    try:
        if not args.skip_smoke_check:
            run_cmd(["python", "-m", "scripts.smoke_check"], allow_fail=args.continue_on_error)

        for layer in layers:
            run_cmd(
                [
                    "python",
                    "-m",
                    "scripts.extract_features",
                    "--config",
                    runtime_config,
                    "--layer",
                    str(layer),
                ],
                allow_fail=args.continue_on_error,
            )

            for seed in seeds:
                for width in widths:
                    run_cmd(
                        [
                            "python",
                            "-m",
                            "scripts.train_layer",
                            "--config",
                            runtime_config,
                            "--layer",
                            str(layer),
                            "--seed",
                            str(seed),
                            "--width",
                            str(width),
                        ],
                        allow_fail=args.continue_on_error,
                    )
                    run_cmd(
                        [
                            "python",
                            "-m",
                            "scripts.evaluate_layer",
                            "--config",
                            runtime_config,
                            "--layer",
                            str(layer),
                            "--seed",
                            str(seed),
                            "--width",
                            str(width),
                        ],
                        allow_fail=args.continue_on_error,
                    )

        if len(seeds) > 1:
            for layer in layers:
                for width in widths:
                    for seed1, seed2 in itertools.combinations(seeds, 2):
                        run_cmd(
                            [
                                "python",
                                "-m",
                                "experiments.run_stability",
                                "--config",
                                runtime_config,
                                "--layer",
                                str(layer),
                                "--width",
                                str(width),
                                "--seed1",
                                str(seed1),
                                "--seed2",
                                str(seed2),
                            ],
                            allow_fail=args.continue_on_error,
                        )

        width_pairs = adjacent_pairs(widths)
        if width_pairs:
            for layer in layers:
                for seed in seeds:
                    for width_small, width_large in width_pairs:
                        run_cmd(
                            [
                                "python",
                                "-m",
                                "experiments.run_cross_width",
                                "--config",
                                runtime_config,
                                "--layer",
                                str(layer),
                                "--seed",
                                str(seed),
                                "--width_small",
                                str(width_small),
                                "--width_large",
                                str(width_large),
                            ],
                            allow_fail=args.continue_on_error,
                        )

        run_cmd(["python", "-m", "experiments.aggregate_results"], allow_fail=args.continue_on_error)
        run_cmd(["python", "-m", "experiments.aggregate_stability"], allow_fail=args.continue_on_error)
        run_cmd(["python", "-m", "experiments.aggregate_cross_width"], allow_fail=args.continue_on_error)

        print("\n=== Run Complete ===")
        print("Generated summaries:")
        for p in ["results_summary.csv", "stability_summary.csv", "cross_width_summary.csv"]:
            print(f"- {p}: {'yes' if os.path.exists(p) else 'no'}")

    finally:
        if using_temp and runtime_config and os.path.exists(runtime_config):
            os.remove(runtime_config)


if __name__ == "__main__":
    main()
