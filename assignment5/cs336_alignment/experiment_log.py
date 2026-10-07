"""Experiment records: metrics, rollouts, checkpoints, and optional wandb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cs336_alignment.checkpoint import save_model_and_tokenizer
from cs336_alignment.training_runtime import TrainAlgorithmConfig


def write_rollouts_jsonl(
    path: Path,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn,
) -> None:
    with path.open("w") as handle:
        for prompt, response, ground_truth in zip(prompts, responses, ground_truths):
            scores = reward_fn(response, ground_truth)
            handle.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "response": response,
                        "ground_truth": ground_truth,
                        **scores,
                    }
                )
                + "\n"
            )


def write_run_config(output_dir, args, algo):
    payload = {}
    cli_config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        cli_config[key] = value
    payload["cli"] = cli_config
    payload["algorithm"] = algo.__dict__
    file_path = output_dir / "run_config.json"
    with file_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def log_step_row(
    output_dir: Path,
    wandb_run: Any,
    row: dict[str, float | int | bool | str],
) -> None:
    row_text = json.dumps(row)
    print(row_text, flush=True)
    metrics_path = output_dir.joinpath("metrics.jsonl")
    with metrics_path.open("a") as handle:
        handle.write(row_text + "\n")
    if wandb_run is not None and int(row["step"]) >= 0:
        import wandb

        wandb.log(row, step=int(row["step"]))


def maybe_init_wandb(args: argparse.Namespace, config: TrainAlgorithmConfig) -> Any:
    if args.wandb_disabled:
        return None
    import wandb

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"{args.recipe}_seed{args.seed}"
    wandb_config = {**vars(args), **config.__dict__}
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=wandb_config,
        dir=str(args.output_dir),
    )


def finish_wandb(wandb_run: Any) -> None:
    if wandb_run is None:
        return
    import wandb

    wandb.finish()


def save_checkpoint(policy, tokenizer, directory: Path) -> None:
    save_model_and_tokenizer(policy, tokenizer, directory)
