"""Plot training curves from results/*/metrics.jsonl (for writeup).

Requires: uv sync --extra plots && uv pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_contiguous_run(rows: list[dict]) -> list[dict]:
    """Keep the suffix after the last decreasing step, so a rerun is not one curve."""
    start = 0
    previous = None
    for index, row in enumerate(rows):
        step = int(row.get("step", 0))
        if previous is not None and step < previous:
            start = index
        previous = step
    return rows[start:]


def plot_runs(run_dirs: list[Path], metric: str, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        rows = [r for r in load_metrics(metrics_path) if r.get("step", 0) >= 0 and metric in r]
        rows = last_contiguous_run(rows)
        if metric == "token_entropy":
            rows = [r for r in rows if not r.get("token_entropy_skipped")]
        if not rows:
            continue
        steps = [int(r["step"]) for r in rows]
        values = [float(r[metric]) for r in rows]
        ax.plot(steps, values, label=run_dir.name, alpha=0.85)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="e.g. results/standard_seed_0 results/standard_seed_1",
    )
    parser.add_argument("--metric", default="val_answer_reward")
    parser.add_argument("--output", type=Path, default=ROOT / "results/plots/metric.png")
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    title = args.title or args.metric
    plot_runs(args.runs, args.metric, args.output, title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
