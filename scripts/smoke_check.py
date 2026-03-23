import importlib
import os
import subprocess
import sys


def check_imports():
    required = ["torch", "torchvision", "timm", "yaml", "pandas", "scipy"]
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def run_cmd(cmd):
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    ok = result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    return ok, output.strip()


def main():
    print(f"Python: {sys.executable}")
    print(f"CWD: {os.getcwd()}")

    missing = check_imports()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
    else:
        print("All core Python packages are importable.")

    checks = [
        ["python", "-m", "scripts.train_layer", "--help"],
        ["python", "-m", "scripts.evaluate_layer", "--help"],
        ["python", "-m", "experiments.run_full_stability_sweep", "--help"],
        ["python", "-m", "experiments.run_cross_width_sweep", "--help"],
        ["python", "-m", "experiments.aggregate_results"],
        ["python", "-m", "experiments.aggregate_stability"],
        ["python", "-m", "experiments.aggregate_cross_width"],
    ]

    for cmd in checks:
        ok, output = run_cmd(cmd)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {' '.join(cmd)}")
        if not ok and output:
            lines = output.splitlines()
            print("  " + lines[-1])

    features_exists = os.path.exists("features/layer_0.pt")
    print(
        "features/layer_0.pt exists: "
        + ("yes" if features_exists else "no (extract features first)")
    )

    print("Smoke check done.")


if __name__ == "__main__":
    main()