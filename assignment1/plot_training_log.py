"""Plot training loss and learning rate from train_lm.py's CSV log."""

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
    parser.add_argument("--output", type=Path, default=Path("training_curve.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iterations: list[int] = []
    losses: list[float] = []
    learning_rates: list[float] = []

    with args.log.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            iterations.append(int(row["iteration"]))
            losses.append(float(row["train_loss"]))
            learning_rates.append(float(row["learning_rate"]))

    figure, loss_axis = plt.subplots(figsize=(8, 5))
    lr_axis = loss_axis.twinx()
    loss_axis.plot(iterations, losses, color="tab:blue", label="train loss")
    lr_axis.plot(iterations, learning_rates, color="tab:orange", label="learning rate")
    loss_axis.set_xlabel("Iteration")
    loss_axis.set_ylabel("Cross-entropy loss", color="tab:blue")
    lr_axis.set_ylabel("Learning rate", color="tab:orange")
    loss_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
