"""Minimal CS336 training loop and one-batch overfitting smoke test.

This example deliberately reuses the functions already connected in
tests/adapters.py.  The only project-specific import that may need changing is
TransformerLM below.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project integration: change this import if your TransformerLM lives elsewhere.
# For example, a root-level transformer.py would use:
#     from transformer import TransformerLM
# ---------------------------------------------------------------------------
from cs336_basics.transformer import TransformerLM

# Because your unit tests pass, these adapters already point to your working
# AdamW, loss, batching, clipping, LR schedule, and checkpoint implementations.
from tests.adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data and outputs.
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--data-dtype", choices=("uint16", "int32"), default="uint16")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/overfit"))
    parser.add_argument("--resume", type=Path, default=None)

    # Small defaults intended for the architecture smoke test.
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=384)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    
    # RMSNorm 消融开关：默认仍使用 RMSNorm
    parser.add_argument(
        "--disable-rmsnorm",
        action="store_true",
        help="跳过所有 TransformerBlock 的 ln1 / ln2，以及模型的 ln_final。",
    )
    
    # ===== [新增：post-norm] 从这里开始添加 =====
    parser.add_argument(
        "--use-post-norm",
        action="store_true",
        help=(
            "将 TransformerBlock 改为子层 -> 残差相加 -> RMSNorm；"
            "保留最终 ln_final。"
        ),
    )
    # ===== [新增：post-norm] 添加结束 =====

    # ===== [新增：NoPE] 开始 =====
    parser.add_argument(
        "--disable-rope",
        action="store_true",
        help=(
            "禁用所有 Attention 中的 RoPE，进行 NoPE 消融；"
            "保留 causal mask、RMSNorm 和 SwiGLU。"
            "不能与 --disable-rmsnorm 或 --use-post-norm 同时使用。"
        ),
    )
    # ===== [新增：NoPE] 结束 =====

    # ===== [新增：SiLU FFN] 开始 =====
    parser.add_argument(
        "--use-silu-ffn",
        action="store_true",
        help=(
            "使用不带门控的 SiLU 前馈网络替代 SwiGLU；"
            "中间维度由 --d-ff 指定。"
            "不能与 --disable-rmsnorm、--use-post-norm "
            "或 --disable-rope 同时使用。"
        ),
    )
    # ===== [新增：SiLU FFN] 结束 =====

    # Optimization.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging and evaluation.
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:0, ...")
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Sample x/y once and reuse exactly that batch for every update.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(args: argparse.Namespace, device: torch.device) -> TransformerLM:
    """Adjust only this call if your constructor has a different signature."""
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
        disable_rmsnorm=args.disable_rmsnorm,  # [新增] 传入消融开关
        use_post_norm=args.use_post_norm,  # [新增：post-norm]
        disable_rope=args.disable_rope,  # [新增：NoPE] 将命令行开关传给模型
        # [新增：SiLU FFN] 将命令行开关传给模型
        use_silu_ffn=args.use_silu_ffn,
    )


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, V)/(B, T) into the 2D/1D adapter interface."""
    vocab_size = logits.shape[-1]
    return run_cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


def gradient_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().pow(2).sum()
    return float(squared_norm.sqrt().item())


@torch.inference_mode()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.memmap,
    args: argparse.Namespace,
    device: torch.device,
    fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    iterations = 1 if fixed_batch is not None else args.eval_iters

    for _ in range(iterations):
        if fixed_batch is None:
            x, y = run_get_batch(
                dataset,
                args.batch_size,
                args.context_length,
                str(device),
            )
        else:
            x, y = fixed_batch
        losses.append(float(lm_loss(model(x), y).item()))

    model.train()
    return sum(losses) / len(losses)


def append_log(
    path: Path,
    *,
    iteration: int,
    train_loss: float,
    validation_loss: float | None,
    learning_rate: float,
    gradient_norm: float,
    elapsed_seconds: float,
) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "iteration",
                "train_loss",
                "validation_loss",
                "learning_rate",
                "gradient_norm_before_clip",
                "elapsed_seconds",
            ),
        )
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "iteration": iteration,
                "train_loss": train_loss,
                "validation_loss": "" if validation_loss is None else validation_loss,
                "learning_rate": learning_rate,
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_seconds": elapsed_seconds,
            }
        )


def main() -> None:
    args = parse_args()
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if args.context_length < 1:
        raise ValueError("context_length must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"

    numpy_dtype = np.dtype(args.data_dtype)
    train_data = np.memmap(args.train_data, dtype=numpy_dtype, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=numpy_dtype, mode="r")
    minimum_length = args.context_length + 1
    if len(train_data) < minimum_length or len(valid_data) < minimum_length:
        raise ValueError(
            f"Each dataset needs at least {minimum_length} tokens; "
            f"got train={len(train_data)}, valid={len(valid_data)}"
        )

    model = build_model(args, device).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        parameters,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iteration = 0
    if args.resume is not None:
        start_iteration = run_load_checkpoint(args.resume, model, optimizer)

    fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    if args.overfit_one_batch:
        fixed_batch = run_get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            str(device),
        )

    parameter_count = sum(parameter.numel() for parameter in parameters)
    print(f"device={device}")
    print(f"parameters={parameter_count:,}")
    print(f"train_tokens={len(train_data):,}; valid_tokens={len(valid_data):,}")
    print(f"overfit_one_batch={args.overfit_one_batch}")

    model.train()
    started_at = time.perf_counter()
    last_validation_loss: float | None = None

    for iteration in range(start_iteration, args.max_iterations):
        learning_rate = run_get_lr_cosine_schedule(
            iteration,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        if fixed_batch is None:
            x, y = run_get_batch(
                train_data,
                args.batch_size,
                args.context_length,
                str(device),
            )
        else:
            x, y = fixed_batch

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lm_loss(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}: {loss.item()}")

        loss.backward()
        gradient_norm = gradient_l2_norm(parameters)
        run_gradient_clipping(parameters, args.max_grad_norm)
        optimizer.step()

        completed = iteration + 1
        should_evaluate = completed == 1 or completed % args.eval_interval == 0
        if should_evaluate:
            # During the smoke test, evaluate the exact batch being memorized.
            # During real training, fixed_batch is None and valid_data is sampled.
            last_validation_loss = estimate_loss(
                model,
                valid_data,
                args,
                device,
                fixed_batch=fixed_batch,
            )

        should_log = completed == 1 or completed % args.log_interval == 0
        if should_log:
            train_loss = float(loss.item())
            elapsed = time.perf_counter() - started_at
            validation_text = (
                "n/a" if last_validation_loss is None else f"{last_validation_loss:.6f}"
            )
            print(
                f"iteration={completed:5d} "
                f"train_loss={train_loss:.6f} "
                f"validation_loss={validation_text} "
                f"lr={learning_rate:.3e} "
                f"grad_norm={gradient_norm:.4f}"
            )
            append_log(
                log_path,
                iteration=completed,
                train_loss=train_loss,
                validation_loss=last_validation_loss if should_evaluate else None,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                elapsed_seconds=elapsed,
            )

        if completed % args.checkpoint_interval == 0:
            checkpoint_path = args.output_dir / f"checkpoint_{completed:06d}.pt"
            run_save_checkpoint(model, optimizer, completed, checkpoint_path)

    final_checkpoint = args.output_dir / "checkpoint_final.pt"
    run_save_checkpoint(model, optimizer, args.max_iterations, final_checkpoint)
    print(f"saved {final_checkpoint}")
    print(f"log {log_path}")


if __name__ == "__main__":
    main()
