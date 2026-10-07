"""Shared GRPO rollout loop used by the §4.3 and later-section training scripts."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.experiment_log import (
    finish_wandb,
    log_step_row,
    maybe_init_wandb,
    save_checkpoint,
    write_rollouts_jsonl,
    write_run_config,
)
from cs336_alignment.prompting_data_construction import (
    examples_from_rows,
    reward_fn_for_prompt,
    vllm_sampling_params,
)
from cs336_alignment.utils import load_data, load_prompt
from cs336_alignment.training_runtime import TrainAlgorithmConfig, run_train_updates, validate_algorithm_config
from cs336_alignment.vllm_utils import VLLMServer


ROOT = Path(__file__).resolve().parents[1]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train", type=Path, default=ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompt", type=Path, default=ROOT / "cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="vLLM sampling temperature (handout §4.3 default 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="vLLM nucleus sampling top_p (handout §4.3 default 1.0)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="vLLM HTTP port; must differ when running multiple jobs in parallel",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-at-start", action="store_true", default=True)
    parser.add_argument("--no-eval-at-start", action="store_false", dest="eval_at_start")
    parser.add_argument("--val-limit", type=int, default=None, help="debug: truncate validation set")
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument(
        "--log-val-rollouts",
        action="store_true",
        default=True,
        help="write val_rollouts_step_*.jsonl whenever validation runs",
    )
    parser.add_argument(
        "--no-log-val-rollouts",
        action="store_false",
        dest="log_val_rollouts",
    )
    parser.add_argument("--save-every", type=int, default=0, help="if >0, save checkpoint every N steps")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-disabled", action="store_true")


def load_examples(path: Path, prompt_path: Path, limit: int) -> list[dict]:
    return examples_from_rows(load_data(path)[:limit], load_prompt(prompt_path))


def take_prompts(examples: list[dict], order: list[int], cursor: int, count: int, rng: random.Random):
    chosen = []
    for _ in range(count):
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        chosen.append(examples[order[cursor]])
        cursor += 1
    return chosen, cursor


def generate_rollout_completions(
    server: VLLMServer,
    prompts: list[str],
    params: dict,
    step: int,
    base_seed: int,
    group_size: int,
) -> list:
    """One vLLM request for every prompt, with n=group_size samples each."""
    roll_params = {
        **params,
        "n": group_size,
        "seed": base_seed + step,
    }
    return server.generate_completions(prompts, roll_params)


def repeat_group(examples: list[dict], group_size: int) -> tuple[list[str], list[str]]:
    prompts = []
    ground_truths = []
    for example in examples:
        prompts.extend([example["prompt"]] * group_size)
        ground_truths.extend([example["ground_truth"]] * group_size)
    return prompts, ground_truths


def evaluate_validation(
    server: VLLMServer,
    examples: list[dict],
    reward_fn,
    params: dict,
    rollout_path: Path | None = None,
) -> dict[str, float]:
    completions = server.generate_completions([example["prompt"] for example in examples], params)
    totals = []
    formats = []
    answers = []
    lengths = []
    for example, completion in zip(examples, completions):
        scores = reward_fn(completion.text, example["ground_truth"])
        totals.append(scores["reward"])
        formats.append(scores["format_reward"])
        answers.append(scores["answer_reward"])
        lengths.append(len(completion.token_ids))
    if rollout_path is not None:
        write_rollouts_jsonl(
            rollout_path,
            [example["prompt"] for example in examples],
            [completion.text for completion in completions],
            [example["ground_truth"] for example in examples],
            reward_fn,
        )
    n = len(examples)
    return {
        "val_reward": sum(totals) / n,
        "val_format_reward": sum(formats) / n,
        "val_answer_reward": sum(answers) / n,
        "val_response_length": sum(lengths) / n,
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args: argparse.Namespace, algo: TrainAlgorithmConfig) -> None:
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")

    validate_algorithm_config(algo, args.rollout_batch_size)
    set_global_seed(args.seed)

    reward_fn = reward_fn_for_prompt(args.prompt.name)
    train_examples = load_examples(args.train, args.prompt, args.n_train_examples)
    val_examples = load_examples(args.val, args.prompt, args.n_val_examples)
    if args.val_limit is not None:
        val_examples = val_examples[: args.val_limit]
    rng = random.Random(args.seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)
    cursor = 0
    prompts_per_step = args.rollout_batch_size // args.group_size

    policy, _ = get_model_and_tokenizer(args.model, args.policy_device)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # One directory is one run. Appending would stitch a rerun onto the old curve.
    (output_dir / "metrics.jsonl").write_text("")
    write_run_config(output_dir, args, algo)
    wandb_run = maybe_init_wandb(args, algo)

    server = VLLMServer(
        model_id=args.model,
        gpu=args.vllm_gpu,
        port=args.vllm_port,
        seed=args.seed,
    )
    server.start()
    server.init_weight_sync(args.policy_device)
    server.sync_policy_weights(policy)
    params = vllm_sampling_params(
        args.prompt.name,
        args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    needs_old_log_probs = algo.importance_reweighting_method != "none"

    try:
        if args.eval_at_start:
            start_row = {"step": -1}
            val_rollout_path = None
            if args.log_val_rollouts:
                val_rollout_path = output_dir / "val_rollouts_step_-1.jsonl"
            start_row.update(
                evaluate_validation(
                    server, val_examples, reward_fn, params, rollout_path=val_rollout_path
                )
            )
            log_step_row(output_dir, wandb_run, start_row)

        for step in range(args.num_rollout_steps):
            batch, cursor = take_prompts(train_examples, order, cursor, prompts_per_step, rng)
            unique_prompts = [example["prompt"] for example in batch]
            prompts, ground_truths = repeat_group(batch, args.group_size)
            server.sync_policy_weights(policy)
            completions = generate_rollout_completions(
                server, unique_prompts, params, step, args.seed, args.group_size
            )
            if len(completions) != len(prompts):
                raise ValueError(
                    "vLLM returned "
                    f"{len(completions)} completions for {len(prompts)} grouped prompts"
                )
            responses = [item.text for item in completions]

            # All-zero advantages still call optimizer.step on explicit zero grads.
            loss, metadata = run_train_updates(
                policy=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                reward_fn=reward_fn,
                prompts=prompts,
                responses=responses,
                ground_truths=ground_truths,
                config=algo,
                group_size=args.group_size,
                max_grad_norm=args.max_grad_norm,
                rng=rng,
                compute_old_log_probs_for_offpolicy=needs_old_log_probs,
            )

            row = {
                "step": step,
                "loss": loss,
                "grad_norm": metadata.get("grad_norm", 0.0),
                "token_entropy": metadata.get("token_entropy", 0.0),
                "token_entropy_skipped": bool(metadata.get("token_entropy_skipped", False)),
                "train_reward": metadata.get("mean_reward", 0.0),
                "train_format_reward": metadata.get("mean_format_reward", 0.0),
                "train_answer_reward": metadata.get("mean_answer_reward", 0.0),
            }
            if "clip_fraction" in metadata:
                row["clip_fraction"] = metadata["clip_fraction"]
            if step % args.eval_every == 0:
                server.sync_policy_weights(policy)
                val_rollout_path = None
                if args.log_val_rollouts:
                    val_rollout_path = output_dir / f"val_rollouts_step_{step}.jsonl"
                row.update(
                    evaluate_validation(
                        server, val_examples, reward_fn, params, rollout_path=val_rollout_path
                    )
                )
            log_step_row(output_dir, wandb_run, row)

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(policy, tokenizer, output_dir / f"checkpoint_step_{step}")

            if step % args.log_rollouts_every == 0:
                write_rollouts_jsonl(
                    output_dir / f"rollouts_step_{step}.jsonl",
                    prompts,
                    responses,
                    ground_truths,
                    reward_fn,
                )
        save_checkpoint(policy, tokenizer, output_dir / "checkpoint_final")
        server.sync_policy_weights(policy)
        final_val_path = None
        if args.log_val_rollouts:
            final_val_path = output_dir / "val_rollouts_final.jsonl"
        final_row = {"step": args.num_rollout_steps, "checkpoint": "final"}
        final_row.update(
            evaluate_validation(
                server, val_examples, reward_fn, params, rollout_path=final_val_path
            )
        )
        log_step_row(output_dir, wandb_run, final_row)
    finally:
        server.stop()
        finish_wandb(wandb_run)
