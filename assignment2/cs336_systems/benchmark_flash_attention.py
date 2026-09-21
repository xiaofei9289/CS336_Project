"""CS336 flash_benchmarking: causal attention on one GPU, batch size 1.

Run from the assignment repository root:
    uv run python -m cs336_systems.benchmark_flash_attention

Default sweep: L=128..65536 (powers of two), d=16/32/64/128,
BF16 and FP32, PyTorch baseline and the assignment FlashAttentionTriton.
The handout specifies a B200; actual GPU and software versions are recorded.

Forward: training-mode attention with autograd tracking enabled.
Backward: reuse ONE output graph with retain_graph=True to time backward alone.
Forward+backward: build a NEW graph every call; retain_graph=False.
Both backward callbacks reset input gradients on every invocation.
No optimizer. All random inputs and the baseline mask are made before timing.
do_bench returns mean CUDA-event milliseconds, not synchronized wall time.
--warmup-ms and --rep-ms are time budgets, NOT iteration counts.

Each (implementation, dtype, L, d) runs in a fresh sequential subprocess.
Successful metrics survive a later OOM; unmeasured metrics remain blank.
CSV has one row per implementation (160 rows for the complete default sweep).
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
    "forward_backward_status", "forward_ms", "backward_ms", "forward_backward_ms",
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
    }


def measure_configuration(torch, do_bench, args, row):
    row["failure_phase"] = "import_attention"
    if row["implementation"] == "pytorch":
        from cs336_basics.model import scaled_dot_product_attention
    else:
        from cs336_systems.flash_attention import FlashAttentionTriton

    row["failure_phase"] = "input_allocation"
    length, dim = row["seq_len"], row["dim"]
    dtype = getattr(torch, row["dtype"])
    shape = (1, length, dim)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    dO = torch.randn(shape, device="cuda", dtype=dtype)

    if row["implementation"] == "pytorch":
        row["failure_phase"] = "mask_allocation"
        # Baseline uses the existing assignment implementation, NOT SDPA/Flash.
        indices = torch.arange(length, device="cuda")
        causal_mask = indices[:, None] >= indices[None, :]
        del indices

        def forward():
            return scaled_dot_product_attention(q, k, v, mask=causal_mask)
    else:
        # Flash builds its causal mask inside each Triton tile.
        def forward():
            return FlashAttentionTriton.apply(q, k, v, True)

    def clear_gradients():
        q.grad = k.grad = v.grad = None

    def record_metric(name, fn):
        row["failure_phase"] = f"{name}_warmup"
        row[f"{name}_status"] = "RUNNING"
        for _ in range(args.warmup_steps):
            fn()
            torch.cuda.synchronize()
        # First Triton/torch.compile executions have finished before do_bench.
        row["failure_phase"] = f"{name}_measurement"
        row[f"{name}_ms"] = float(do_bench(
            fn, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="mean"
        ))
        row[f"{name}_status"] = "OK"

    torch.cuda.synchronize()
    record_metric("forward", forward)
    clear_gradients()
    torch.cuda.synchronize()

    row["failure_phase"] = "backward_prepare"
    row["backward_status"] = "RUNNING"
    out = forward()
    torch.cuda.synchronize()

    def backward():
        clear_gradients()
        out.backward(dO, retain_graph=True)

    record_metric("backward", backward)
    # Release the retained graph BEFORE benchmarking fresh forward+backward.
    del backward, out
    clear_gradients()
    torch.cuda.synchronize()

    def forward_backward():
        clear_gradients()
        output = forward()
        output.backward(dO)

    # This is measured directly, not computed by adding two mean latencies.
    record_metric("forward_backward", forward_backward)


def run_worker(args):
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
        measure_configuration(torch, do_bench, args, row)
        row.update(status="OK", failure_phase="")
    except Exception as exc:
        oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        row["status"] = "OOM" if oom else "ERROR"
        row["error"] = str(exc) if oom else traceback.format_exc()
        for name in ("forward", "backward", "forward_backward"):
            if row[f"{name}_status"] == "RUNNING":
                row[f"{name}_status"] = row["status"]

    # CPU-only write after an error; process exit releases all GPU allocations.
    args.result_json.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return int(row["status"] == "ERROR")


def run_parent(args):
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd()), *[p for p in sys.path if p], env.get("PYTHONPATH", "")]
    )
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
                            row = new_row(args, implementation, dtype, length, dim)
                            result_path = Path(temp) / f"{implementation}_{dtype}_{length}_{dim}.json"
                            command = [
                                sys.executable, str(Path(__file__).resolve()), "--worker",
                                "--result-json", str(result_path), "--implementations", implementation,
                                "--dtypes", dtype, "--seq-lengths", str(length), "--dims", str(dim),
                                "--warmup-steps", str(args.warmup_steps), "--warmup-ms", str(args.warmup_ms),
                                "--rep-ms", str(args.rep_ms), "--seed", str(args.seed),
                            ]
                            print(f"Running {implementation}: {dtype}, L={length}, d={dim}", flush=True)
                            child = subprocess.run(command, env=env, capture_output=True, text=True)
                            if result_path.exists():
                                row = json.loads(result_path.read_text(encoding="utf-8"))
                            else:
                                row.update(
                                    status="ERROR", failure_phase="child_process",
                                    error=child.stderr[-8000:] or "Child exited without a result",
                                )
                            row["child_returncode"] = child.returncode
                            if child.returncode != 0:
                                row["status"] = "ERROR"
                            writer.writerow(row)
                            file.flush()
                            print(
                                f"  {row['status']} | fwd={row.get('forward_ms', 'NA')} ms"
                                f" | bwd={row.get('backward_ms', 'NA')} ms"
                                f" | fwd+bwd={row.get('forward_backward_ms', 'NA')} ms"
                                f" | phase={row['failure_phase']}", flush=True,
                            )
                            if row["status"] == "ERROR":
                                failed = True
                                print(row.get("error", "Child failed"), file=sys.stderr)
                                if row["failure_phase"] in {"setup", "import_attention"}:
                                    return 1
    print(f"Saved: {args.output}")
    return int(failed)


if __name__ == "__main__":
    arguments = parse_args()
    sys.exit(run_worker(arguments) if arguments.worker else run_parent(arguments))
