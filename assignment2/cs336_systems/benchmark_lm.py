"""CS336 assignment 2 language-model benchmark.

File layout (read in this order; it matches the run):
  1. Constants     — Table 1 sizes and mode labels
  2. parse_args    — CLI, presets
  3. One training step — forward / loss / backward / optimizer, with timing
  4. Warmup check  — first-step sanity checks (not official timing)
  5. Reporting     — stats, stdout, optional JSON / memory snapshot
  6. main          — CUDA, build model, warmup, measure, report

Run inside the assignment repository (which supplies torch and cs336_basics):
  uv run python -m cs336_systems.benchmark_lm --model-size small --mode train --timing stages
  uv run python -m cs336_systems.benchmark_lm --model-size small --mode train --timing total
  uv run python -m cs336_systems.benchmark_lm --model-size small --mode forward --inference

For torch_compile(b), compile the ENTIRE BasicsTransformerLM with --compile:
  uv run python -m cs336_systems.benchmark_lm --model-size small --mode forward --timing total --compile
Repeat with --mode forward_backward and --mode train, then with medium.
Defaults: batch=4, context=512, sequence length equals context, warmup=5, measurement=10, FP32,
torch.optim.AdamW for train. Do not enable inference unless measuring forward-only.
Use --sequence-length to run a shorter batch than the model context window.
Whole-model compilation uses torch.compile(model) with default settings,
not fullgraph=True and not max-autotune. The optimizer is not compiled.
The same-mode warmup includes backward/optimizer as applicable, so initial
compilation and AdamW state initialization are excluded from timing.
--compile automatically writes a new results/torch_compile_*_compiled.json;
use --results-json for another filename if a previous result already exists.
Use the same hardware/software and TF32 settings for eager comparisons.
TF32 settings are inherited (as before) and recorded, not silently changed.
For nsys_profile, add --nvtx and capture the measurement range:
  nsys profile --trace cuda,nvtx --capture-range nvtx --nvtx-capture measurement \\
    -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 -- python -m cs336_systems.benchmark_lm --nvtx ...

For gradient_checkpointing(b), wrap consecutive blocks after the model is on CUDA.
Do not change BasicsTransformerLM.forward. Leave --compile and --inference off:
  python -m cs336_systems.benchmark_lm --model-size xl --batch-size 4 --context-length 2048 \\
    --mode forward_backward --precision fp32 --checkpoint-size 1 \\
    --warmup-steps 1 --measurement-steps 1 \\
    --results-json results/checkpoint_xl_b4_ctx2048_fp32_fwd_bwd_k1.json
Repeat with --checkpoint-size 2 and 4. Peak memory is max_memory_allocated during measurement.
"""

import argparse
from contextlib import nullcontext
import importlib
import json
import math
from pathlib import Path
import statistics
from timeit import default_timer

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F
from einops import einsum
from cs336_basics import model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Handout Table 1: (d_model, d_ff, num_layers, num_heads)
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}

MODE_MEASUREMENT_NAMES = {
    "forward": "forward",
    "forward_backward": "forward + loss + backward",
    "train": "forward + loss + backward + optimizer (excluding zero_grad)",
}


# ---------------------------------------------------------------------------
# 2. parse_args
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI flags and fill Table-1 presets."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward")
    parser.add_argument("--timing", choices=["total", "stages"], default="total")
    parser.add_argument("--model-size", choices=list(MODEL_SIZES), default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="RoPE / constructor maximum sequence length",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Actual input length; defaults to --context-length",
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference", action="store_true", help="Disable gradient tracking; forward mode only")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--memory-snapshot",
        type=Path,
        help="Record CUDA memory from before model.to('cuda') through measurement, then save a snapshot",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the entire BasicsTransformerLM with default torch.compile settings",
    )
    parser.add_argument("--results-json", type=Path, help="Write configuration and metrics to a new JSON file")
    parser.add_argument(
        "--optimizer-class",
        help="Import AdamW as module.path:ClassName; default torch.optim.AdamW",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX ranges: warmup, measurement, forward, loss, backward, optimizer, attention",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=0,
        help="Consecutive TransformerBlocks per non-nested checkpoint; 0 leaves layers unchanged",
    )
    args = parser.parse_args()

    args.d_model, args.d_ff, args.num_layers, args.num_heads = MODEL_SIZES[args.model_size]
    if args.checkpoint_size < 0 or args.checkpoint_size > args.num_layers:
        parser.error("--checkpoint-size must be between 0 and the model's layer count")
    if args.checkpoint_size and args.inference:
        parser.error("--checkpoint-size requires gradient tracking; omit --inference")
    if args.checkpoint_size and args.compile:
        parser.error("checkpoint comparison is eager; omit --compile")
    if args.sequence_length is None:
        args.sequence_length = args.context_length
    if args.compile and args.results_json is None:
        forward_kind = "inference" if args.inference else "training"
        seq_tag = (
            f"_seq{args.sequence_length}"
            if args.sequence_length != args.context_length
            else ""
        )
        args.results_json = Path(
            f"results/torch_compile_{args.model_size}_b{args.batch_size}"
            f"_ctx{args.context_length}{seq_tag}_{args.precision}_{args.mode}"
            f"_{args.timing}_{forward_kind}_compiled.json"
        )
    return args


# ---------------------------------------------------------------------------
# 3. One training step (the timed unit)
# ---------------------------------------------------------------------------


def unwrap_model(model):
    """Return the original nn.Module if `model` is a torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


def install_checkpoint_groups(model, checkpoint_size):
    """Replace model.layers with non-nested groups of `checkpoint_size` blocks.

    k=0 does nothing. Each group runs its original blocks in order and calls
    checkpoint once when grad is enabled. A remainder slice becomes the last group.
    """
    if checkpoint_size == 0:
        return

    from torch.utils.checkpoint import checkpoint

    class CheckpointGroup(torch.nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = torch.nn.ModuleList(blocks)

        def run_blocks(self, hidden):
            for block in self.blocks:
                hidden = block(hidden)
            return hidden

        def forward(self, hidden):
            if torch.is_grad_enabled():
                return checkpoint(self.run_blocks, hidden, use_reentrant=False)
            return self.run_blocks(hidden)

    blocks = list(model.layers)
    model.layers = torch.nn.ModuleList(
        CheckpointGroup(blocks[start:start + checkpoint_size])
        for start in range(0, len(blocks), checkpoint_size)
    )


def autocast_context(precision):
    """BF16 autocast when --precision bf16; otherwise a no-op context manager."""
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def nvtx_range(args, name):
    """NVTX range when --nvtx is set; otherwise a no-op."""
    if args.nvtx:
        return nvtx.range(name)
    return nullcontext()


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Same math as cs336_basics.model.scaled_dot_product_attention, with NVTX ranges."""
    with nvtx.range("scaled_dot_product_attention"):
        d_k = K.shape[-1]
        with nvtx.range("scores_matmul"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with nvtx.range("attention_softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with nvtx.range("attention_final_matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def measure_cuda(fn):
    """Time `fn` in milliseconds with CUDA synchronize before and after."""
    torch.cuda.synchronize()
    start = default_timer()
    result = fn()
    torch.cuda.synchronize()
    return result, (default_timer() - start) * 1000


def run_one_step(args, model, x, y, optimizer, split=False):
    """Run one step for --mode. If split=True, time forward/loss/backward/optimizer separately."""
    timings = {}
    loss = None

    def phase(name, fn):
        if split:
            result, timings[name] = measure_cuda(fn)
            return result
        return fn()

    def forward():
        with torch.set_grad_enabled(not args.inference), autocast_context(args.precision):
            return model(x)

    with nvtx_range(args, "forward"):
        logits = phase("forward", forward)
    if args.mode != "forward":
        def compute_loss():
            with autocast_context(args.precision):
                return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        with nvtx_range(args, "loss"):
            loss = phase("loss", compute_loss)
        # Backward and optimizer stay outside autocast.
        with nvtx_range(args, "backward"):
            phase("backward", loss.backward)
        if args.mode == "train":
            with nvtx_range(args, "optimizer"):
                phase("optimizer", optimizer.step)
    return timings, logits, loss


def timed_step(args, model, x, y, optimizer):
    """Time one step as --timing total (one number) or stages (per-phase numbers)."""
    if args.timing == "total":
        (_, logits, loss), elapsed_ms = measure_cuda(
            lambda: run_one_step(args, model, x, y, optimizer)
        )
        return {"total": elapsed_ms}, logits, loss
    timings, logits, loss = run_one_step(args, model, x, y, optimizer, split=True)
    return timings, logits, loss


def format_step(timings):
    """Format a timings dict as a single log line."""
    return " ".join(f"{name}={value:.3f}ms" for name, value in timings.items())


# ---------------------------------------------------------------------------
# 4. Warmup check
# ---------------------------------------------------------------------------

def check_first_step(logits, loss, model, args, param_before):
    """Validate logits/loss/grads (and a train-mode param update) after the first warmup step."""

    expected = (args.batch_size, args.sequence_length, args.vocab_size)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"logits shape {tuple(logits.shape)} != {expected}")
    if args.inference:
        if logits.requires_grad:
            raise RuntimeError("logits must not require grad under --inference")
        return
    if not logits.requires_grad:
        raise RuntimeError("training-forward logits must require grad")
    if args.mode == "forward":
        return
    if loss is None or loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"invalid loss: {loss}")
    weight = unwrap_model(model).lm_head.weight
    if weight.grad is None or weight.grad.shape != weight.shape:
        raise RuntimeError("lm_head.weight.grad missing or has the wrong shape after backward")
    if not torch.isfinite(weight.grad).all():
        raise RuntimeError("lm_head.weight.grad contains non-finite values")
    if args.mode != "train":
        return
    after = weight.detach()
    if param_before is None:
        return
    if not torch.isfinite(after).all():
        raise RuntimeError("parameters contain non-finite values after the optimizer step")
    if torch.equal(after, param_before):
        raise RuntimeError("lm_head.weight did not change in train mode")
    max_change = (after - param_before).abs().max().item()
    print(f"first warmup: lm_head.weight changed; max abs delta={max_change:.8e}")


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------

def summarize_ms(values):
    """Reduce a list of millisecond samples to mean, population std, min, and max."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def print_run_header(args, model, optimizer, execution_mode):
    """Print resolved config before warmup so a later OOM still leaves a log."""
    size_label = args.model_size
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"execution: {execution_mode}")
    print(f"TF32: matmul={torch.backends.cuda.matmul.allow_tf32}; cudnn={torch.backends.cudnn.allow_tf32}")
    print(f"measurement: {MODE_MEASUREMENT_NAMES[args.mode]}  |  timing={args.timing}")
    print(
        f"model size: {size_label}; structure: d_model={args.d_model} d_ff={args.d_ff} "
        f"num_layers={args.num_layers} num_heads={args.num_heads}"
    )
    print(f"context_length (RoPE max)={args.context_length}; sequence_length (input)={args.sequence_length}")
    print(f"batch_size={args.batch_size}; vocab_size={args.vocab_size}")
    print(f"precision: FP32 parameters / {'BF16 autocast' if args.precision == 'bf16' else 'FP32 compute'}")
    print(f"forward kind: {'inference (grad tracking off)' if args.inference else 'training forward (grad tracking on)'}")
    print(f"parameter count: {sum(p.numel() for p in model.parameters()):,}")
    if optimizer is not None:
        group = optimizer.param_groups[0]
        print(f"optimizer: {type(optimizer).__module__}.{type(optimizer).__name__}")
        print(f"lr={group.get('lr')}; weight_decay={group.get('weight_decay')}")
    print("Timing excludes input generation, model init, and zero_grad. Std is pstdev.")
    if args.timing == "stages":
        print("CUDA is synchronized between stages; do not treat the stage sum as end-to-end time.")
    if args.memory_snapshot:
        print("Memory snapshot is enabled; reported times include that overhead. Run a separate unprofiled baseline.")
    if args.compile:
        print("Compile target: entire BasicsTransformerLM; default torch.compile; optimizer not compiled.")
    if args.nvtx:
        print("NVTX ranges: warmup / measurement / forward / loss / backward / optimizer / attention.")
    if args.checkpoint_size:
        print(
            f"checkpoint: {args.checkpoint_size} consecutive blocks per group, "
            "one non-nested checkpoint when grad is enabled"
        )
        if args.mode == "forward":
            print("checkpoint peak includes recomputation only in backward; use --mode forward_backward")
    if args.warmup_steps == 0:
        print("warmup-steps=0: skipping first-step correctness checks.")
        if args.mode == "train":
            print("warmup-steps=0 and mode=train: the first measured step includes optimizer-state initialization.")


def print_and_save_results(
    args,
    model,
    optimizer,
    execution_mode,
    records,
    baseline_bytes,
    peak_allocated,
    peak_reserved,
):
    """Print timing/memory summary and optionally write JSON / snapshot path."""
    print("\n--- measurement summary ---")
    for name, values in records.items():
        stats = summarize_ms(values)
        print(f"\n{name}: {[round(t, 3) for t in values]} ms")
        print(
            f"mean={stats['mean']:.3f} ms; std={stats['std']:.3f} ms; "
            f"min={stats['min']:.3f} ms; max={stats['max']:.3f} ms"
        )
    print(f"\nallocated before measurement: {baseline_bytes / 1024**2:.2f} MiB")
    print(f"peak allocated during measurement: {peak_allocated / 1024**2:.2f} MiB")
    print(f"peak reserved during measurement: {peak_reserved / 1024**2:.2f} MiB")
    if args.results_json:
        result = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "execution_mode": execution_mode,
            "compile_scope": "entire_model" if args.compile else None,
            "compile_settings": (
                {"backend": "inductor", "mode": "default", "fullgraph": False}
                if args.compile else None
            ),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "zero_grad_in_timing": False,
            "measurement_name": MODE_MEASUREMENT_NAMES[args.mode],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "optimizer": None if optimizer is None else f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            "baseline_allocated_mib": baseline_bytes / 1024**2,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "peak_reserved_mib": peak_reserved / 1024**2,
            "timings_ms": records,
            "timing_summary_ms": {name: summarize_ms(values) for name, values in records.items()},
            "memory_scope": "Total CUDA allocator memory during measurement, including backward when enabled; not activation-only",
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        if args.results_json.exists():
            print(f"Results JSON already exists, skipped: {args.results_json}")
        else:
            with args.results_json.open("x") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results JSON: {args.results_json}")
    if args.memory_snapshot:
        print(f"Memory snapshot: {args.memory_snapshot}")


# ---------------------------------------------------------------------------
# 6. main — orchestration only
# ---------------------------------------------------------------------------

def build_model_and_batch(args):
    """Construct the LM, optional compile, random batch, and optimizer."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    # Start before .to("cuda") so parameter allocations keep Python stacks.
    # Do not cap max_entries: PyTorch drops the oldest events first, which are these weights.
    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history()
    model = model.to(device="cuda", dtype=torch.float32)
    model.train(not args.inference)
    install_checkpoint_groups(model, args.checkpoint_size)

    if args.nvtx:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # Default torch.compile: no fullgraph, no max-autotune; loss and optimizer stay eager.
    # Actual compilation happens on the matching-mode warmup in main().
    if args.compile:
        model = torch.compile(model)
    execution_mode = "compiled_model" if args.compile else "eager"

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device="cuda")
    y = None
    if args.mode != "forward":
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device="cuda")
    optimizer = None
    if args.mode == "train":
        optimizer_cls = torch.optim.AdamW
        if args.optimizer_class:
            module_name, class_name = args.optimizer_class.split(":", 1)
            optimizer_cls = getattr(importlib.import_module(module_name), class_name)
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    return model, optimizer, x, y, execution_mode


def run_warmup(args, model, x, y, optimizer):
    """Run the same mode/timing path as measurement; first step is sanity-checked."""
    param_before = None
    if args.mode == "train" and args.warmup_steps > 0:
        param_before = unwrap_model(model).lm_head.weight.detach().clone()

    with nvtx_range(args, "warmup"):
        for i in range(args.warmup_steps):
            model.zero_grad(set_to_none=True)
            timings, logits, loss = timed_step(args, model, x, y, optimizer)
            print(f"warmup {i + 1}/{args.warmup_steps}: {format_step(timings)}")
            if i == 0:
                check_first_step(logits, loss, model, args, param_before)
                print("first warmup check passed")
                param_before = None
            del logits, loss


def run_measurement(args, model, x, y, optimizer):
    """Timed loop. zero_grad is outside the timed region. Returns records and peak memory."""
    torch.cuda.synchronize()
    if args.memory_snapshot:
        args.memory_snapshot.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    records = {}

    try:
        with nvtx_range(args, "measurement"):
            for i in range(args.measurement_steps):
                model.zero_grad(set_to_none=True)
                step_timings, logits, loss = timed_step(args, model, x, y, optimizer)
                del logits, loss
                print(f"measurement {i + 1}/{args.measurement_steps}: {format_step(step_timings)}")
                for name, value in step_timings.items():
                    records.setdefault(name, []).append(value)
        model.zero_grad(set_to_none=True)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        if args.memory_snapshot:
            torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
    finally:
        if args.memory_snapshot:
            torch.cuda.memory._record_memory_history(enabled=None)
    return records, baseline_bytes, peak_allocated, peak_reserved


def main():
    """Parse args, build the LM, warm up, time the requested mode, print results."""
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision fp32")

    model, optimizer, x, y, execution_mode = build_model_and_batch(args)
    print_run_header(args, model, optimizer, execution_mode)
    run_warmup(args, model, x, y, optimizer)
    records, baseline_bytes, peak_allocated, peak_reserved = run_measurement(
        args, model, x, y, optimizer
    )
    print_and_save_results(
        args,
        model,
        optimizer,
        execution_mode,
        records,
        baseline_bytes,
        peak_allocated,
        peak_reserved,
    )


if __name__ == "__main__":
    main()
