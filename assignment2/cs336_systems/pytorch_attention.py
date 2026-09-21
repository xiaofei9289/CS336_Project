"""pytorch_attention / torch_compile (a): isolated CUDA benchmarks.

Run from the assignment repository root:
    python -m cs336_systems.pytorch_attention
    python -m cs336_systems.pytorch_attention --compile

Requires the existing cs336_basics.model.scaled_dot_product_attention.
No explicit FlashAttention, head dimension, causal mask, or optimizer.
--compile compiles the same attention function with TorchInductor, fullgraph=True.
The first forward AND backward compilations occur in warmup, outside reported
timings. Inputs, precision, and the measurement loop are identical in both modes.
Times are synchronized wall-clock milliseconds. Memory is live allocated
tensor memory before backward, NOT reserved memory or peak memory.
Each child exits after one configuration; the parent never initializes CUDA.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from timeit import default_timer
import traceback


LENGTHS = [256, 1024, 4096, 8192, 16384]
DIMS = [16, 32, 64, 128]
MIB = 1024 ** 2
IMPLEMENTATION = "cs336_basics.model.scaled_dot_product_attention"
FIELDS = [
    "implementation", "execution_mode", "compile_backend", "compile_fullgraph",
    "gpu", "gpu_memory_gib", "torch_version", "cuda_version",
    "dtype", "tf32", "batch_size", "seq_len", "dim", "causal", "seed",
    "warmup", "iterations", "status", "failure_phase", "failure_iteration",
    "forward_samples", "backward_samples", "forward_mean_ms", "forward_std_ms",
    "backward_mean_ms", "backward_std_ms", "baseline_allocated_mib",
    "before_backward_allocated_mib", "memory_source", "memory_samples",
    "forward_live_extra_mib", "error", "child_returncode",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the existing attention with TorchInductor; warm up forward and backward")
    parser.add_argument("--output", type=Path, default=None)
    # Filters are useful for smoke tests and fresh-process OOM reproduction.
    parser.add_argument("--seq-lengths", nargs="+", type=int, choices=LENGTHS, default=LENGTHS)
    parser.add_argument("--dims", nargs="+", type=int, choices=DIMS, default=DIMS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must both be positive")
    if args.worker and (len(args.seq_lengths) != 1 or len(args.dims) != 1 or args.result_json is None):
        parser.error("worker requires exactly one configuration and result-json")
    if args.output is None:
        prefix = "compiled_attention" if args.compile else "pytorch_attention"
        args.output = Path(f"results/{prefix}_{args.dtype}.csv")
    return args


def measure_configuration(torch, attention, args, row, forward_ms, backward_ms, memory):
    """Mutate CPU-only records so an OOM does not discard earlier measurements."""
    dtype = getattr(torch, args.dtype)
    row["failure_phase"] = "input_allocation"
    shape = (8, row["seq_len"], row["dim"])
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    grad_output = torch.randn(shape, device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    row["baseline_allocated_mib"] = torch.cuda.memory_allocated() / MIB

    for stage, count in (("warmup", args.warmup), ("measurement", args.iterations)):
        for index in range(count):
            row["failure_iteration"] = index + 1
            row["failure_phase"] = f"{stage}_prepare"
            q.grad = k.grad = v.grad = None
            torch.cuda.synchronize()

            row["failure_phase"] = f"{stage}_forward"
            start = default_timer()
            out = attention(q, k, v, mask=None)
            torch.cuda.synchronize()
            elapsed = (default_timer() - start) * 1000
            if stage == "measurement":
                forward_ms.append(elapsed)

            row["failure_phase"] = f"{stage}_memory"
            allocated = torch.cuda.memory_allocated() / MIB
            # Record warmup memory too: the first backward may already OOM.
            memory[stage].append(allocated)

            row["failure_phase"] = f"{stage}_backward"
            start = default_timer()
            out.backward(grad_output)
            torch.cuda.synchronize()
            elapsed = (default_timer() - start) * 1000
            if stage == "measurement":
                backward_ms.append(elapsed)
            del out


def summarize(row, forward_ms, backward_ms, memory):
    # Partial timing means remain available but MUST be read with status/counts.
    for name, samples in (("forward", forward_ms), ("backward", backward_ms)):
        row[f"{name}_samples"] = len(samples)
        if samples:
            row[f"{name}_mean_ms"] = statistics.mean(samples)
            row[f"{name}_std_ms"] = statistics.pstdev(samples)
    source = "measurement" if memory["measurement"] else "warmup"
    samples = memory[source]
    row["memory_samples"] = len(samples)
    if samples:
        mean = statistics.mean(samples)
        row["before_backward_allocated_mib"] = mean
        row["memory_source"] = source
        row["forward_live_extra_mib"] = mean - row["baseline_allocated_mib"]


def run_worker(args):
    row = {
        "implementation": IMPLEMENTATION, "dtype": args.dtype, "tf32": False,
        "execution_mode": "compiled" if args.compile else "eager",
        "compile_backend": "inductor" if args.compile else "",
        "compile_fullgraph": True if args.compile else "",
        "batch_size": 8, "seq_len": args.seq_lengths[0], "dim": args.dims[0],
        "causal": False, "seed": args.seed, "warmup": args.warmup,
        "iterations": args.iterations, "status": "ERROR", "failure_phase": "setup",
    }
    forward_ms, backward_ms = [], []
    memory = {"warmup": [], "measurement": []}
    torch = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on this GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        row.update(
            gpu=torch.cuda.get_device_name(0),
            gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        )
        row["failure_phase"] = "import_attention"
        # Verified against the uploaded model(1).py. Do not silently substitute
        # another implementation if the assignment package is missing.
        from cs336_basics.model import scaled_dot_product_attention

        attention = scaled_dot_product_attention
        if args.compile:
            row["failure_phase"] = "compile_setup"
            # Fail visibly instead of silently falling back to eager on errors.
            # Compilation is lazy: first calls in warmup compile both passes.
            torch._dynamo.config.suppress_errors = False
            attention = torch.compile(attention, backend="inductor", fullgraph=True)

        measure_configuration(
            torch, attention, args, row,
            forward_ms, backward_ms, memory,
        )
        row.update(status="OK", failure_phase="", failure_iteration="")
    except Exception as exc:
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            row["status"] = "OOM"
            row["error"] = str(exc)
        else:
            row["status"] = "ERROR"
            row["error"] = traceback.format_exc()

    # No further CUDA calls after failure. Only CPU summaries and file writes;
    # process exit releases this configuration's CUDA context and allocations.
    summarize(row, forward_ms, backward_ms, memory)
    args.result_json.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 1 if row["status"] == "ERROR" else 0


def run_parent(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Protect existing measurements; choose a new --output for another run.
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose another --output")
    env = os.environ.copy()
    # Preserve repository imports when launching this script by absolute path.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd()), *[p for p in sys.path if p], env.get("PYTHONPATH", "")]
    )
    script = str(Path(__file__).resolve())
    failed = False
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        with tempfile.TemporaryDirectory(prefix="attention_bench_") as temp:
            for dim in args.dims:
                for length in args.seq_lengths:
                    result_path = Path(temp) / f"d{dim}_L{length}.json"
                    command = [
                        sys.executable, script, "--worker", "--result-json", str(result_path),
                        "--dims", str(dim), "--seq-lengths", str(length),
                        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                        "--dtype", args.dtype, "--seed", str(args.seed),
                    ]
                    if args.compile:
                        command.append("--compile")
                    mode = "compiled" if args.compile else "eager"
                    print(f"Running {mode}, L={length}, d={dim}, dtype={args.dtype}", flush=True)
                    child = subprocess.run(command, env=env, capture_output=True, text=True)
                    if result_path.exists():
                        row = json.loads(result_path.read_text(encoding="utf-8"))
                    else:
                        # A killed/crashed process is not automatically an OOM.
                        row = {
                            "seq_len": length, "dim": dim, "dtype": args.dtype,
                            "execution_mode": mode,
                            "status": "ERROR", "failure_phase": "child_process",
                            "error": child.stderr[-8000:] or "Child produced no result",
                        }
                    row["child_returncode"] = child.returncode
                    if child.returncode != 0:
                        row["status"] = "ERROR"
                    writer.writerow(row)
                    file.flush()
                    print(
                        f"  {row['status']} | forward={row.get('forward_mean_ms', 'NA')} ms"
                        f" | backward={row.get('backward_mean_ms', 'NA')} ms"
                        f" | before backward={row.get('before_backward_allocated_mib', 'NA')} MiB"
                        f" | phase={row.get('failure_phase', '')}", flush=True,
                    )
                    if row["status"] == "ERROR":
                        print(row.get("error", "Child process failed"), file=sys.stderr)
                        failed = True
                        # Missing dependencies/setup errors affect every configuration.
                        if row.get("failure_phase") in {"setup", "import_attention"}:
                            return 1
    print(f"Saved: {args.output}")
    return int(failed)


if __name__ == "__main__":
    parsed = parse_args()
    sys.exit(run_worker(parsed) if parsed.worker else run_parent(parsed))
