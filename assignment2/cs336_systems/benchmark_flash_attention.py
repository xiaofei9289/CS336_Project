"""Benchmark causal attention: eager PyTorch vs partial Triton FlashAttention.

For every dtype, embedding size, and sequence length:
    1. Create random Q, K, V and an output gradient.
    2. Time the forward pass.
    3. Time the backward pass on one saved graph.
    4. Time a fresh forward followed by its backward.
    5. Write one CSV row.

Sequence lengths are powers of two from 128 through 65536.
Embedding sizes are 16, 32, 64, and 128. Precisions are BF16 and FP32.
Batch size is 1, and attention is causal. There is no optimizer.

Each (implementation, dtype, length, d) runs in its own process so one
out-of-memory failure cannot disturb the next configuration. The process
you launch only queues those runs and writes the CSV.

    python -m cs336_systems.benchmark_flash_attention
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


LENGTHS = [2 ** power for power in range(7, 17)]
DIMS = [16, 32, 64, 128]
DTYPES = ["bfloat16", "float32"]
IMPLEMENTATIONS = ["pytorch", "flash"]
FIELDS = [
    "implementation", "forward_implementation", "backward_implementation",
    "gpu", "gpu_memory_gib", "torch_version", "cuda_version", "triton_version",
    "batch_size", "seq_len", "dim", "dtype", "causal", "tf32_matmul", "seed",
    "warmup_steps", "warmup_ms", "rep_ms", "timing_method", "backward_method",
    "status", "failure_phase", "forward_status", "backward_status",
    "forward_backward_status", "q_tile", "k_tile",
    "forward_ms", "backward_ms", "forward_backward_ms",
    "error", "child_returncode",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq-lengths", nargs="+", type=int, choices=LENGTHS, default=LENGTHS)
    parser.add_argument("--dims", nargs="+", type=int, choices=DIMS, default=DIMS)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=DTYPES)
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=IMPLEMENTATIONS)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Untimed calls before do_bench, including lazy compilation")
    parser.add_argument("--warmup-ms", type=float, default=25,
                        help="Additional do_bench warmup duration in milliseconds")
    parser.add_argument("--rep-ms", type=float, default=100,
                        help="do_bench measurement duration in milliseconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/flash_benchmarking.csv"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.warmup_ms <= 0 or args.rep_ms <= 0:
        parser.error("warmup-steps, warmup-ms and rep-ms must be positive")
    if args.worker and (
        args.result_json is None or any(len(getattr(args, name)) != 1 for name in
                                       ("seq_lengths", "dims", "dtypes", "implementations"))
    ):
        parser.error("A worker requires exactly one configuration and --result-json")
    return args


def new_row(args, implementation, dtype, length, dim):
    return {
        "implementation": implementation,
        "forward_implementation": (
            "cs336_basics.model.scaled_dot_product_attention" if implementation == "pytorch"
            else "cs336_systems.flash_attention.FlashAttentionTriton"
        ),
        "backward_implementation": (
            "torch_autograd" if implementation == "pytorch" else "compiled_pytorch_recomputation"
        ),
        "batch_size": 1, "seq_len": length, "dim": dim, "dtype": dtype,
        "causal": True, "tf32_matmul": False, "seed": args.seed,
        "warmup_steps": args.warmup_steps, "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms, "timing_method": "triton.testing.do_bench_mean_ms",
        "backward_method": "single_retained_graph; reset_input_grads_each_call",
        "status": "ERROR", "failure_phase": "setup",
        "forward_status": "NOT_RUN", "backward_status": "NOT_RUN",
        "forward_backward_status": "NOT_RUN",
        "q_tile": "", "k_tile": "",
    }


def prepare_config(torch, row):
    """Import the selected attention and allocate this configuration's inputs."""
    row["failure_phase"] = "import_attention"
    if row["implementation"] == "pytorch":
        from cs336_basics.model import scaled_dot_product_attention
    else:
        from cs336_systems.flash_attention import FlashAttentionTriton, choose_flash_tiles

    row["failure_phase"] = "input_allocation"
    length, dim = row["seq_len"], row["dim"]
    dtype = getattr(torch, row["dtype"])
    shape = (1, length, dim)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    # backward() needs an explicit gradient because the attention output is not a scalar.
    dO = torch.randn(shape, device="cuda", dtype=dtype)

    if row["implementation"] == "pytorch":
        row["failure_phase"] = "mask_allocation"
        # The assignment attention takes an explicit mask. Flash builds causality inside its kernel.
        indices = torch.arange(length, device="cuda")
        causal_mask = indices[:, None] >= indices[None, :]
        del indices

        def forward():
            return scaled_dot_product_attention(q, k, v, mask=causal_mask)
    else:
        q_tile, k_tile = choose_flash_tiles(length, length, dim)
        row["q_tile"] = q_tile
        row["k_tile"] = k_tile

        def forward():
            return FlashAttentionTriton.apply(q, k, v, True)

    return q, k, v, dO, forward


def time_metric(torch, do_bench, args, row, name, fn):
    """Warm up fn, then record its mean do_bench latency in milliseconds."""
    row["failure_phase"] = f"{name}_warmup"
    row[f"{name}_status"] = "RUNNING"
    for _ in range(args.warmup_steps):
        fn()
        torch.cuda.synchronize()
    # First Triton or torch.compile executions have finished before do_bench.
    # The assignment asks for triton.testing.do_bench, which times CUDA events.
    row["failure_phase"] = f"{name}_measurement"
    row[f"{name}_ms"] = float(do_bench(
        fn, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="mean"
    ))
    row[f"{name}_status"] = "OK"


def measure_config(args):
    """Child process: time one configuration and write one JSON row."""
    row = new_row(args, args.implementations[0], args.dtypes[0], args.seq_lengths[0], args.dims[0])
    torch = None
    try:
        import torch
        import triton
        from triton.testing import do_bench

        if not torch.cuda.is_available():
            raise RuntimeError("This benchmark requires a CUDA GPU")
        if row["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16")
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        row.update(
            gpu=torch.cuda.get_device_name(0),
            gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            triton_version=str(triton.__version__),
        )
        q, k, v, dO, forward = prepare_config(torch, row)

        def clear_gradients():
            # Drop saved gradients so repeated backwards do not accumulate.
            q.grad = k.grad = v.grad = None

        torch.cuda.synchronize()
        time_metric(torch, do_bench, args, row, "forward", forward)
        clear_gradients()
        torch.cuda.synchronize()

        row["failure_phase"] = "backward_prepare"
        row["backward_status"] = "RUNNING"
        out = forward()
        torch.cuda.synchronize()

        def backward():
            clear_gradients()
            # Keep this graph so backward can be repeated without another forward.
            out.backward(dO, retain_graph=True)

        time_metric(torch, do_bench, args, row, "backward", backward)
        # Drop the saved graph before timing a fresh forward together with its backward.
        del backward, out
        clear_gradients()
        torch.cuda.synchronize()

        def forward_backward():
            clear_gradients()
            output = forward()
            output.backward(dO)

        # Measured directly. Do not add the forward mean to the backward mean.
        time_metric(torch, do_bench, args, row, "forward_backward", forward_backward)
        row.update(status="OK", failure_phase="")
    except Exception as exc:
        oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        row["status"] = "OOM" if oom else "ERROR"
        row["error"] = str(exc) if oom else traceback.format_exc()
        for name in ("forward", "backward", "forward_backward"):
            if row[f"{name}_status"] == "RUNNING":
                row[f"{name}_status"] = row["status"]
        # Forward+backward includes backward. If backward already ran out of
        # memory, that combined measurement cannot succeed.
        if (
            oom
            and row["backward_status"] == "OOM"
            and row["forward_backward_status"] == "NOT_RUN"
        ):
            row["forward_backward_status"] = "OOM"

    # CPU-only write after an error; process exit releases all GPU allocations.
    args.result_json.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return int(row["status"] == "ERROR")


def launch_one(args, script, env, result_path, implementation, dtype, length, dim):
    """Start one child for a single configuration and wait until it exits."""
    command = [
        sys.executable, script, "--worker", "--result-json", str(result_path),
        "--implementations", implementation, "--dtypes", dtype,
        "--seq-lengths", str(length), "--dims", str(dim),
        "--warmup-steps", str(args.warmup_steps), "--warmup-ms", str(args.warmup_ms),
        "--rep-ms", str(args.rep_ms), "--seed", str(args.seed),
    ]
    return subprocess.run(command, env=env, capture_output=True, text=True)


def read_one(child, result_path, row):
    """Turn one finished child into a CSV row."""
    if result_path.exists():
        row = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        # A killed process may leave no JSON. That is a crash, not an OOM record.
        row.update(
            status="ERROR", failure_phase="child_process",
            error=child.stderr[-8000:] or "Child exited without a result",
        )
    row["child_returncode"] = child.returncode
    if child.returncode != 0:
        row["status"] = "ERROR"
    return row


def prepare_run(args):
    """Create the result directory and the environment shared by every child."""
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Preserve repository imports when launching this script by absolute path.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd()), *[p for p in sys.path if p], env.get("PYTHONPATH", "")]
    )
    script = str(Path(__file__).resolve())
    return script, env


def record_config(args, file, writer, script, env, result_path, implementation, dtype, length, dim):
    """Measure one configuration and append its row. Return 'ok', 'error', or 'stop'."""
    print(f"Running {implementation}: {dtype}, L={length}, d={dim}", flush=True)
    row = new_row(args, implementation, dtype, length, dim)
    child = launch_one(args, script, env, result_path, implementation, dtype, length, dim)
    row = read_one(child, result_path, row)
    writer.writerow(row)
    file.flush()
    print(
        f"  {row['status']} | fwd={row.get('forward_ms', 'NA')} ms"
        f" | bwd={row.get('backward_ms', 'NA')} ms"
        f" | fwd+bwd={row.get('forward_backward_ms', 'NA')} ms"
        f" | phase={row['failure_phase']}", flush=True,
    )
    if row["status"] != "ERROR":
        return "ok"
    print(row.get("error", "Child failed"), file=sys.stderr)
    # Missing dependencies or a missing attention import affect every configuration.
    if row["failure_phase"] in {"setup", "import_attention"}:
        return "stop"
    return "error"


def run_all(args):
    """Queue every configuration in its own process and write the CSV.

    A separate process is what releases that configuration's GPU memory before
    the next one starts. This process does not call CUDA.
    """
    script, env = prepare_run(args)
    failed = False
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        with tempfile.TemporaryDirectory(prefix="flash_bench_") as temp:
            for dtype in args.dtypes:
                for dim in args.dims:
                    for length in args.seq_lengths:
                        for implementation in args.implementations:
                            result_path = Path(temp) / f"{implementation}_{dtype}_{length}_{dim}.json"
                            outcome = record_config(
                                args, file, writer, script, env, result_path,
                                implementation, dtype, length, dim,
                            )
                            if outcome == "stop":
                                return 1
                            if outcome == "error":
                                failed = True
    print(f"Saved: {args.output}")
    return int(failed)


def main():
    args = parse_args()
    if args.worker:
        sys.exit(measure_config(args))
    sys.exit(run_all(args))


if __name__ == "__main__":
    main()
