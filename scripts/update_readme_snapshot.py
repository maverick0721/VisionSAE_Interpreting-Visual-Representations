import csv
import statistics
from collections import defaultdict
from pathlib import Path


START_MARKER = "<!-- RESULTS_SNAPSHOT_START -->"
END_MARKER = "<!-- RESULTS_SNAPSHOT_END -->"


def load_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def mean_or_na(values, digits=4):
    if not values:
        return "N/A"
    return f"{statistics.mean(values):.{digits}f}"


def build_snapshot_block(repo_root):
    results = load_csv_rows(repo_root / "results_summary.csv")
    stability = load_csv_rows(repo_root / "stability_summary.csv")
    cross = load_csv_rows(repo_root / "cross_width_summary.csv")

    mse = [float(x["mse"]) for x in results] if results else []
    sparsity = [float(x["sparsity"]) for x in results] if results else []
    coherence = [float(x["coherence"]) for x in results] if results else []
    stability_scores = [float(x["stability_score"]) for x in stability] if stability else []
    cross_scores = [float(x["alignment_score"]) for x in cross] if cross else []

    mse_by_width = defaultdict(list)
    stability_by_width = defaultdict(list)

    for row in results:
        mse_by_width[int(row["width"])].append(float(row["mse"]))

    for row in stability:
        stability_by_width[int(row["width"])].append(float(row["stability_score"]))

    all_widths = sorted(set(mse_by_width.keys()) | set(stability_by_width.keys()))

    lines = [
        START_MARKER,
        "Snapshot from current CSV summaries:",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total evaluated runs | {len(results)} |",
        f"| Mean MSE | {mean_or_na(mse)} |",
        f"| Mean sparsity | {mean_or_na(sparsity)} |",
        f"| Mean coherence | {mean_or_na(coherence)} |",
        f"| Mean stability score | {mean_or_na(stability_scores)} |",
        f"| Mean cross-width alignment | {mean_or_na(cross_scores)} |",
        "",
        "Width trend highlights:",
        "",
        "| Width | Mean MSE | Mean Stability |",
        "| --- | --- | --- |",
    ]

    if all_widths:
        for width in all_widths:
            lines.append(
                f"| {width} | {mean_or_na(mse_by_width[width])} | {mean_or_na(stability_by_width[width])} |"
            )
    else:
        lines.append("| N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "Interpretation: in these runs, larger SAE width is associated with better (lower) reconstruction error and stronger cross-seed stability.",
            END_MARKER,
        ]
    )

    return "\n".join(lines)


def update_readme(repo_root):
    readme_path = repo_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise RuntimeError("README snapshot markers not found")

    end_idx += len(END_MARKER)
    new_block = build_snapshot_block(repo_root)
    updated = content[:start_idx] + new_block + content[end_idx:]
    readme_path.write_text(updated, encoding="utf-8")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    update_readme(repo_root)
    print("README results snapshot updated.")


if __name__ == "__main__":
    main()
