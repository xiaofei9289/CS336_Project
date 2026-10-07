"""DPO on single-turn Anthropic HH. Does not use Hugging Face Trainer.

The policy and the frozen reference sit on separate devices. Validation
accuracy follows the handout: a pair is correct when the policy assigns
higher log-probability to the chosen completion than to the rejected one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.optim import RMSprop
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.dpo import per_instance_dpo_loss, response_log_prob
from cs336_alignment.grad_accum import apply_partial_window_scale
from cs336_alignment.hh_data import load_hh_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="instruction-tuned checkpoint")
    parser.add_argument("--hh-dir", type=Path, default=Path("data/hh"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--ref-device", default="cuda:1")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_hh_train(args.hh_dir)
    rng.shuffle(rows)
    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size :]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.policy_device)
    reference = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.ref_device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    optimizer = RMSprop(policy.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "dpo_metrics.jsonl"
    best_accuracy = -1.0
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    window_loss = 0.0
    pending = 0
    n_optimizer_steps = math.ceil(len(train_rows) / args.gradient_accumulation_steps)

    def validate(step: int) -> float:
        policy.eval()
        correct = 0
        with torch.no_grad():
            for row in val_rows:
                chosen = response_log_prob(policy, tokenizer, row["instruction"], row["chosen"])
                rejected = response_log_prob(policy, tokenizer, row["instruction"], row["rejected"])
                correct += int(chosen > rejected)
        accuracy = correct / max(1, len(val_rows))
        policy.train()
        with log_path.open("a") as handle:
            handle.write(json.dumps({"step": step, "val_accuracy": accuracy}) + "\n")
        return accuracy

    def save_if_best(accuracy: float) -> None:
        nonlocal best_accuracy
        if accuracy <= best_accuracy:
            return
        best_accuracy = accuracy
        save_dir = args.output_dir / "best"
        policy.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    for index, row in enumerate(train_rows):
        loss = per_instance_dpo_loss(
            policy,
            reference,
            tokenizer,
            args.beta,
            row["instruction"],
            row["chosen"],
            row["rejected"],
        )
        (loss / args.gradient_accumulation_steps).backward()
        window_loss += float(loss.detach())
        pending += 1
        is_last = index + 1 == len(train_rows)
        if pending != args.gradient_accumulation_steps and not is_last:
            continue
        apply_partial_window_scale(policy, args.gradient_accumulation_steps, pending)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step = (index + 1 + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        mean_loss = window_loss / pending
        window_loss = 0.0
        pending = 0
        with log_path.open("a") as handle:
            handle.write(json.dumps({"step": step, "train_loss": mean_loss}) + "\n")
        if step % args.val_every == 0 or step == n_optimizer_steps:
            save_if_best(validate(step))
    print(f"best_val_accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
