"""Plot train/validation loss against iteration and wall-clock time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the step curve; defaults to the log directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Train: keep every row from the CSV.
    iterations: list[int] = []
    elapsed_seconds: list[float] = []
    train_losses: list[float] = []

    # Val: keep only rows with a non-empty validation_loss, and their x coordinates.
    val_iterations: list[int] = []
    val_elapsed_seconds: list[float] = []
    val_losses: list[float] = []

    with args.log.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            iteration = int(row["iteration"])
            elapsed = float(row["elapsed_seconds"])

            iterations.append(iteration)
            elapsed_seconds.append(elapsed)
            train_losses.append(float(row["train_loss"]))

            # Skip empty strings instead of converting them to 0.
            validation_loss = row["validation_loss"].strip()
            if validation_loss:
                val_iterations.append(iteration)
                val_elapsed_seconds.append(elapsed)
                val_losses.append(float(validation_loss))

    # By default, write next to the CSV.
    step_output = (
        args.output
        if args.output is not None
        else args.log.parent / "training_curve_step.png"
    )
    wallclock_output = args.log.parent / "training_curve_wallclock.png"

    # Both figures share the plotting logic; only the x-axis and output path change.
    plots = [
        (
            iterations,
            val_iterations,
            "Iteration",
            step_output,
        ),
        (
            elapsed_seconds,
            val_elapsed_seconds,
            "Wall-clock time (s)",
            wallclock_output,
        ),
    ]

    for train_x, val_x, xlabel, output in plots:
        figure, axis = plt.subplots(figsize=(8, 5))

        axis.plot(
            train_x,
            train_losses,
            color="tab:blue",
            label="Train loss",
            linewidth=1,
        )

        if val_losses:
            axis.plot(
                val_x,
                val_losses,
                color="tab:orange",
                label="Val loss",
                marker="o",
                markersize=3,
                linewidth=1.5,
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel("Cross-entropy loss")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)
        print(f"saved {output}")


if __name__ == "__main__":
    main()