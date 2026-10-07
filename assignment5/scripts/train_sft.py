"""Instruction-tune a causal LM. Does not use Hugging Face Trainer.

Each microbatch loss is weighted by its sequence count, then divided by the
number of sequences in the optimizer step. A window of sizes 2, 2, 1 matches
one batch of those 5 sequences.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.grad_accum import average_sequence_grads
from cs336_alignment.sft_data import PackedSFTDataset


def learning_rate(step: int, total_steps: int, base_lr: float, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def language_model_loss(model, batch, device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def run_epoch(model, tokenizer, data_path: Path, seq_length: int, batch_size: int, device, shuffle: bool) -> float:
    dataset = PackedSFTDataset(tokenizer, data_path, seq_length, shuffle=shuffle)
    weighted = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            n = batch["input_ids"].shape[0]
            weighted += float(language_model_loss(model, batch, device)) * n
            count += n
    model.train()
    return weighted / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    dataset = PackedSFTDataset(tokenizer, args.train, args.seq_length, shuffle=True)
    steps_per_epoch = math.ceil(len(dataset) / args.microbatch_size)
    optimizer_steps = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_index = 0
    opt_step = 0
    pending = 0
    running = 0.0
    window_sequences = 0
    log_path = args.output_dir / "sft_metrics.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for batch in DataLoader(dataset, batch_size=args.microbatch_size, shuffle=True):
        n = batch["input_ids"].shape[0]
        lr = learning_rate(opt_step, optimizer_steps, args.learning_rate, args.warmup_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = language_model_loss(model, batch, device)
        (loss * n).backward()
        running += float(loss.detach()) * n
        window_sequences += n
        micro_index += 1
        pending += 1
        is_last = micro_index == steps_per_epoch
        if pending != args.gradient_accumulation_steps and not is_last:
            continue
        average_sequence_grads(model, window_sequences)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = running / window_sequences
        running = 0.0
        window_sequences = 0
        pending = 0
        with log_path.open("a") as handle:
            handle.write(
                json_line(
                    {
                        "optimizer_step": opt_step,
                        "train_loss": mean_loss,
                        "learning_rate": lr,
                    }
                )
            )
        opt_step += 1

    val_loss = run_epoch(
        model, tokenizer, args.val, args.seq_length, args.microbatch_size, device, shuffle=False
    )
    with log_path.open("a") as handle:
        handle.write(json_line({"val_loss": val_loss, "optimizer_step": opt_step}))
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"val_loss={val_loss:.4f} saved={args.output_dir}")


def json_line(row: dict) -> str:
    import json

    return json.dumps(row) + "\n"


if __name__ == "__main__":
    main()
