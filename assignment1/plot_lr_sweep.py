"""Compare train/validation loss across multiple training runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # nargs="+"：允许 --log 后面传入多个 CSV 路径。
    parser.add_argument(
        "--log",
        type=Path,
        nargs="+",
        required=True,
    )

    # 标签按顺序对应各份 CSV；不传时使用 CSV 的父目录名。
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the step comparison.",
    )

    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic loss axis.",
    )

    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.log):
        parser.error("--labels 的数量必须与 --log 的文件数量一致")

    return args


def read_log(path: Path) -> dict:
    """读取一份 CSV，分别保存 train 和 validation 数据。"""
    data = {
        "iterations": [],
        "elapsed_seconds": [],
        "train_losses": [],
        "val_iterations": [],
        "val_elapsed_seconds": [],
        "val_losses": [],
    }

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            iteration = int(row["iteration"])
            elapsed = float(row["elapsed_seconds"])

            data["iterations"].append(iteration)
            data["elapsed_seconds"].append(elapsed)
            data["train_losses"].append(float(row["train_loss"]))

            # 保留原逻辑：仅读取 validation_loss 非空的行。
            validation_loss = row["validation_loss"].strip()
            if validation_loss:
                data["val_iterations"].append(iteration)
                data["val_elapsed_seconds"].append(elapsed)
                data["val_losses"].append(float(validation_loss))

    return data


def main() -> None:
    args = parse_args()

    labels = (
        args.labels
        if args.labels is not None
        else [path.parent.name for path in args.log]
    )

    # 每份 CSV 只读取一次。
    runs = [read_log(path) for path in args.log]

    step_output = (
        args.output
        if args.output is not None
        else args.log[0].parent / "lr_sweep_step.png"
    )

    # 时间图与步数图保存到同一个目录，避免覆盖单次 run 的图。
    wallclock_output = step_output.with_name(
        f"{step_output.stem}_wallclock{step_output.suffix}"
    )

    plots = [
        (
            "iterations",
            "val_iterations",
            "Training steps",
            step_output,
        ),
        (
            "elapsed_seconds",
            "val_elapsed_seconds",
            "Wall-clock time (s)",
            wallclock_output,
        ),
    ]

    colors = plt.get_cmap("tab10").colors

    for train_key, val_key, xlabel, output in plots:
        # 每张图创建两个子图，使用相同的纵轴范围。
        figure, (train_axis, val_axis) = plt.subplots(
            1, 2,
            figsize=(13, 5),
            sharex=True,
            sharey=True,
        )

        # 每个 run 都向两个子图各添加一条曲线。
        for index, (data, label) in enumerate(zip(runs, labels)):
            color = colors[index % len(colors)]

            train_axis.plot(
                data[train_key],
                data["train_losses"],
                color=color,
                label=label,
                linewidth=1,
                alpha=0.85,
            )

            if data["val_losses"]:
                val_axis.plot(
                    data[val_key],
                    data["val_losses"],
                    color=color,
                    label=label,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                )

        train_axis.set_title("Training loss")
        val_axis.set_title("Validation loss")

        for axis in (train_axis, val_axis):
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Cross-entropy loss")
            axis.grid(alpha=0.25)

            if args.log_y:
                axis.set_yscale("log")

            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()

        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200)
        plt.close(figure)

        print(f"saved {output}")


if __name__ == "__main__":
    main()