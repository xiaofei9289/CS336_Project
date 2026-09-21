# 【新增 2026-09-17】下方补充第三部分 (b) 的运行说明：固定配置比较 checkpoint 分组。
# 原因：本轮实验保留原 xl 的 32 头配置并默认关闭编译，只改变分组大小以保证可比性。
"""CS336 assignment 2 benchmark.

Run inside the assignment repository (which supplies torch and cs336_basics):
  uv run python benchmark.py --model-size small --mode train --timing stages
  uv run python benchmark.py --model-size small --mode train --timing total
  uv run python benchmark.py --model-size small --mode forward --inference

For torch_compile(b), compile the ENTIRE BasicsTransformerLM with --compile:
  uv run python -m cs336_systems.benchmark --model-size small --mode forward --timing total --compile
Repeat with --mode forward_backward and --mode train, then with medium.
Defaults: batch=4, context=512, warmup=5, measurement=10, FP32,
torch.optim.AdamW for train. Do not enable checkpointing, NVTX or inference.
Whole-model compilation uses torch.compile(model) with default settings,
not fullgraph=True and not max-autotune. The optimizer is not compiled.
The same-mode warmup includes backward/optimizer as applicable, so initial
compilation and AdamW state initialization are excluded from timing.
--compile automatically writes a new results/torch_compile_*_compiled.json;
use --results-json for another filename if a previous result already exists.
Use the same hardware/software and TF32 settings for eager comparisons.
TF32 settings are inherited (as before) and recorded, not silently changed.

For gradient_checkpointing(b), use --model-size xl (32 heads by default)
--batch-size 4 --context-length 2048 --mode forward_backward --precision fp32
--checkpoint-size 1 (compare 2, then further neighbors if needed).
Leave --compile-blocks OFF for the initial checkpoint comparison.
--results-json PATH saves metrics and actual configuration.
Only this benchmark needs updating: checkpoint groups wrap the original layers
without changing the BasicsTransformerLM constructor or forward source.
For (c), launch a fresh process for each --warmup-steps 0/1/2/5.
For mixed precision, add --precision bf16 (parameters stay FP32).
For Nsight, add --nvtx and filter to the "measurement" range; use total timing.
For memory, add --memory-snapshot snapshots/xl_train_512.pickle.
For nsys_profile(d), supply your actual assignment-1 AdamW class using
--optimizer-class your_package.your_module:AdamW (no silent fallback).
With --nvtx, attention is replaced in memory with an equivalent annotated
implementation matching the supplied model.py; the model source is not edited.
Inside attention_scores, use scores_matmul for pure matmul kernel attribution.
NVTX CPU range widths are not GPU runtimes; inspect associated CUDA kernels.
Profiling has overhead: run plain timing and profiling as separate processes.
"""

import argparse
from contextlib import nullcontext
import importlib
# 【新增 2026-09-17】导入 JSON 模块，保存可复查的配置、显存数据和计时结果。
import json
from pathlib import Path
import statistics
from timeit import default_timer


MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}


def make_annotated_attention(model_module):
    """Copy the supplied attention computation, adding only NVTX ranges.

    Reuse the model's einsum and custom softmax (not torch.softmax).
    No synchronization is added; gradients and autocast follow the caller.
    """
    torch = model_module.torch
    einsum = model_module.einsum
    softmax = model_module.softmax
    math = model_module.math

    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        with torch.cuda.nvtx.range("scaled_dot_product_attention"):
            d_k = K.shape[-1]
            with torch.cuda.nvtx.range("attention_scores"):
                with torch.cuda.nvtx.range("scores_matmul"):
                    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
                with torch.cuda.nvtx.range("scores_scale"):
                    attention_scores = attention_scores / math.sqrt(d_k)
                if mask is not None:
                    with torch.cuda.nvtx.range("scores_mask"):
                        attention_scores = torch.where(mask, attention_scores, float("-inf"))
            with torch.cuda.nvtx.range("attention_softmax"):
                attention_weights = softmax(attention_scores, dim=-1)
            with torch.cuda.nvtx.range("attention_final_matmul"):
                return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return annotated_scaled_dot_product_attention



# 【修改 2026-09-17】在 benchmark 内将原有层包装成非嵌套检查点分组。
# 原因：兼容作业原版模型接口，无需给构造函数增加参数或改动模型 forward。
def install_checkpoint_groups(model, checkpoint_size):
    if checkpoint_size == 0:
        return  # 不替换 layers，完全保留原来的逐层执行路径。

    # 【新增 2026-09-17】延迟导入，使 --help 和参数校验仍可在无 PyTorch 环境下运行。
    import torch
    from torch.utils.checkpoint import checkpoint

    class CheckpointGroup(torch.nn.Module):
        def __init__(self, blocks):
            super().__init__()
            # 注册原来的模块对象，保留参数、设备和梯度连接，不复制模型权重。
            self.blocks = torch.nn.ModuleList(blocks)

        def run_blocks(self, hidden):
            # 每段内部仅顺序执行原始 block，不创建任何嵌套 checkpoint。
            for block in self.blocks:
                hidden = block(hidden)
            return hidden

        def forward(self, hidden):
            # 保存本段入口张量；反向需要时重算本段所需的前向计算。
            # 非重入实现可能在取齐所需张量后提前结束重算，并非保证重跑每个算子。
            if torch.is_grad_enabled():
                return checkpoint(self.run_blocks, hidden, use_reentrant=False)
            return self.run_blocks(hidden)

    blocks = list(model.layers)
    # 【新增 2026-09-17】切片自动处理余数：32 层、k=3 时最后一段为 2 层。
    # 原模型的 for layer in self.layers 现在依次调用各组，每组只套一次 checkpoint。
    # 包装只用于本次 benchmark；模块路径会变化，不用此结构导出训练权重文件。
    model.layers = torch.nn.ModuleList(
        CheckpointGroup(blocks[start:start + checkpoint_size])
        for start in range(0, len(blocks), checkpoint_size)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward")
    parser.add_argument("--timing", choices=["total", "stages"], default="total")
    parser.add_argument("--model-size", choices=list(MODEL_SIZES), default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    # Explicit dimension arguments override the selected preset.
    for name in ["d-model", "d-ff", "num-layers", "num-heads"]:
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference", action="store_true", help="Disable gradient tracking; forward mode only")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--nvtx", action="store_true", help="Label warmup, measurement and individual phases")
    parser.add_argument("--memory-snapshot", type=Path, help="Save CUDA memory history after measurement")
    # 【新增 2026-09-17】通过命令行切换连续层分组大小，便于独立进程比较 k=1、2 等候选。
    parser.add_argument("--checkpoint-size", type=int, default=0,
                        help="Consecutive layers per non-nested checkpoint; 0 disables")
    # 【新增 2026-09-17】提供逐层编译开关以对齐讲义示例；各对照组必须采用同一设置。
    parser.add_argument("--compile-blocks", action="store_true",
                        help="Compile each TransformerBlock with fullgraph=True")
    # 第四章 torch_compile(b)：编译完整模型，不复用逐层编译开关。
    parser.add_argument("--compile", action="store_true",
                        help="Compile the entire BasicsTransformerLM with default torch.compile settings")
    # 【新增 2026-09-17】提供结果输出路径，避免只依靠终端日志手工整理实验表格。
    parser.add_argument("--results-json", type=Path, help="Write configuration and measured metrics to a new JSON file")
    parser.add_argument("--optimizer-class", help="Import your AdamW using module.path:ClassName; default: torch.optim.AdamW")
    args = parser.parse_args()

    for name, value in zip(["d_model", "d_ff", "num_layers", "num_heads"], MODEL_SIZES[args.model_size]):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.warmup_steps < 0 or args.measurement_steps < 1:
        parser.error("warmup-steps must be >= 0 and measurement-steps must be >= 1")
    if any(getattr(args, name) <= 0 for name in ["batch_size", "context_length", "vocab_size", "d_model", "d_ff", "num_layers", "num_heads"]):
        parser.error("All model dimensions and batch size must be positive")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d_model must be divisible by num_heads; RoPE head dimension must be even")
    if args.inference and args.mode != "forward":
        parser.error("--inference requires --mode forward")
    if args.lr <= 0 or args.rope_theta <= 0:
        parser.error("lr and rope-theta must be positive")
    if args.optimizer_class and args.mode != "train":
        parser.error("--optimizer-class requires --mode train")
    if args.optimizer_class and (":" not in args.optimizer_class or not all(args.optimizer_class.split(":", 1))):
        parser.error("--optimizer-class must have the form module.path:ClassName")
    if args.memory_snapshot and args.memory_snapshot.exists():
        parser.error("Snapshot already exists; choose a new filename")
    # 【新增 2026-09-17】提前拒绝无效分组大小，避免初始化大模型后才发现配置错误。
    if not 0 <= args.checkpoint_size <= args.num_layers:
        parser.error("checkpoint-size must be between 0 and num-layers")
    # 【新增 2026-09-17】推理不保存反向计算图，不能用于验证 checkpoint 的训练显存收益。
    if args.checkpoint_size and args.inference:
        parser.error("checkpointing requires gradient tracking; remove --inference")
    # 【新增 2026-09-17】分开编译实验和 NVTX 注释实验，避免本脚本的标注逻辑干扰 fullgraph 编译。
    if args.compile_blocks and args.nvtx:
        parser.error("Run --compile-blocks and --nvtx separately")
    if args.compile and args.compile_blocks:
        parser.error("--compile and --compile-blocks are mutually exclusive")
    if args.compile and (args.checkpoint_size or args.nvtx):
        parser.error("For whole-model compilation, use --checkpoint-size 0 and omit --nvtx")
    if args.compile and args.warmup_steps < 1:
        parser.error("--compile requires warmup-steps >= 1 to exclude initial compilation")
    if args.compile and args.results_json is None:
        forward_kind = "inference" if args.inference else "training"
        args.results_json = Path(
            f"results/torch_compile_{args.model_size}_b{args.batch_size}"
            f"_ctx{args.context_length}_{args.precision}_{args.mode}"
            f"_{args.timing}_{forward_kind}_compiled.json"
        )
    # 【新增 2026-09-17】拒绝覆盖已有结果，防止重复实验丢失先前测量数据。
    if args.results_json and args.results_json.exists():
        parser.error("Results file already exists; choose a new filename")
    return args


def main():
    args = parse_args()
    # Delayed imports allow --help and configuration checks without CUDA packages.
    import torch
    import torch.nn.functional as F
    import cs336_basics.model as model_module
    from cs336_basics.model import BasicsTransformerLM

    if not torch.cuda.is_available():
        raise RuntimeError("此脚本需要可用的 CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16，请使用 --precision fp32")

    # The supplied attention forward resolves this module global on each call.
    # Patch only profiling runs; normal benchmarks retain the original function.
    if args.nvtx:
        model_module.scaled_dot_product_attention = make_annotated_attention(model_module)

    torch.manual_seed(args.seed)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device="cuda", dtype=torch.float32)
    # 【新增 2026-09-17】按需编译每个 TransformerBlock；编译在后续 warmup 中触发。
    # 原因：保留可选编译实验；先编译原始 block，再分组，避免把分组包装器一起编译。
    # 第三部分 (b) 的初次比较不传 --compile-blocks，各组保持相同设置。
    if args.compile_blocks:
        for i, block in enumerate(model.layers):
            model.layers[i] = torch.compile(block, fullgraph=True)
    # 【修改 2026-09-17】模型上 GPU 后、创建优化器前安装分组，原构造函数不接收 checkpoint 参数。
    install_checkpoint_groups(model, args.checkpoint_size)
    model.train(not args.inference)

    # 包装完整模型一次；前向和反向的实际编译在下方同模式 warmup 中触发。
    # 不强制 fullgraph，不使用 max-autotune，不编译 loss 或 optimizer。
    if args.compile:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = False
        model = torch.compile(model)
    execution_mode = (
        "compiled_model" if args.compile
        else "compiled_blocks" if args.compile_blocks
        else "eager"
    )

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
    optimizer = None
    if args.mode == "train":
        optimizer_cls = torch.optim.AdamW
        if args.optimizer_class:
            module_name, class_name = args.optimizer_class.split(":", 1)
            optimizer_cls = getattr(importlib.import_module(module_name), class_name)
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    def nvtx_range(name):
        return torch.cuda.nvtx.range(name) if args.nvtx else nullcontext()

    def autocast_context():
        if args.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward():
        with torch.set_grad_enabled(not args.inference), autocast_context():
            return model(x)

    def compute_loss(logits):
        with autocast_context():
            return F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

    def measure(fn):
        torch.cuda.synchronize()
        start = default_timer()
        result = fn()
        torch.cuda.synchronize()
        return result, (default_timer() - start) * 1000

    def run_step(split=False):
        timings = {}

        def phase(name, fn):
            with nvtx_range(name):
                if split:
                    result, timings[name] = measure(fn)
                    return result
                return fn()

        logits = phase("forward", forward)
        if args.mode != "forward":
            loss = phase("loss", lambda: compute_loss(logits))
            # Backward and optimizer run outside autocast.
            phase("backward", loss.backward)
            if args.mode == "train":
                phase("optimizer", optimizer.step)
        return timings

    # Warm up the SAME mode and timing path, but discard the timings.
    with nvtx_range("warmup"):
        for _ in range(args.warmup_steps):
            model.zero_grad(set_to_none=True)
            run_step(split=args.timing == "stages")
            torch.cuda.synchronize()

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if args.memory_snapshot:
        args.memory_snapshot.parent.mkdir(parents=True, exist_ok=True)
        # These private APIs are the APIs used in the assignment handout.
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    records = {}

    try:
        with nvtx_range("measurement"):
            for _ in range(args.measurement_steps):
                # Excluded from timing, but visible in the memory/Nsight trace.
                with nvtx_range("zero_grad"):
                    model.zero_grad(set_to_none=True)
                if args.timing == "total":
                    _, elapsed_ms = measure(run_step)
                    step_timings = {"total": elapsed_ms}
                else:
                    step_timings = run_step(split=True)
                torch.cuda.synchronize()
                for name, value in step_timings.items():
                    records.setdefault(name, []).append(value)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        if args.memory_snapshot:
            torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
    finally:
        if args.memory_snapshot:
            torch.cuda.memory._record_memory_history(enabled=None)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"执行方式: {execution_mode}")
    print(f"TF32: matmul={torch.backends.cuda.matmul.allow_tf32}; cudnn={torch.backends.cudnn.allow_tf32}")
    if args.compile:
        print("编译对象: 整个 BasicsTransformerLM；默认 torch.compile 设置；优化器未编译。")
    print(f"配置（实际参数）: {vars(args)}")
    print(f"精度: FP32 参数 / {'BF16 autocast' if args.precision == 'bf16' else 'FP32 运算'}")
    print(f"前向类型: {'推理（关闭梯度追踪）' if args.inference else '训练前向（开启梯度追踪）'}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    if optimizer is not None:
        print(f"优化器: {type(optimizer).__module__}.{type(optimizer).__name__}")
    print("计时不包含输入生成、模型初始化、清梯度；标准差使用 pstdev。")
    if args.timing == "stages":
        print("各阶段之间有 CUDA 同步；阶段之和不作为端到端耗时。")
    if args.nvtx or args.memory_snapshot:
        print("已启用 profiling，耗时包含其额外开销；请另跑无 profiling 的基准。")
    for name, values in records.items():
        print(f"\n{name}: {[round(t, 3) for t in values]} ms")
        print(f"mean={statistics.mean(values):.3f} ms; std={statistics.pstdev(values):.3f} ms")
    print(f"\n测量前 allocated: {baseline_bytes / 1024**2:.2f} MiB")
    print(f"测量期间 peak allocated: {peak_allocated / 1024**2:.2f} MiB")
    print(f"测量期间 peak reserved: {peak_reserved / 1024**2:.2f} MiB")
    # 【新增 2026-09-17】提醒仅前向测量遗漏反向重算峰值，不能单独用来回答第三部分 (b)。
    if args.checkpoint_size and args.mode == "forward":
        print("注意：仅前向不能验证 checkpoint 重算时的峰值；(b) 请用 forward_backward 或 train。")
    # 【新增 2026-09-17】保存实际配置、设备、参数量、显存及计时，便于对比相邻分组并复现实验。
    # 原因：这里复用原脚本测量值；peak allocated 是总分配显存，不应误标成纯激活显存。
    if args.results_json:
        result = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "execution_mode": execution_mode,
            "compile_scope": "entire_model" if args.compile else "blocks" if args.compile_blocks else None,
            "compile_settings": (
                {"backend": "inductor", "mode": "default", "fullgraph": False}
                if args.compile else
                {"backend": "inductor", "mode": "default", "fullgraph": True}
                if args.compile_blocks else None
            ),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "zero_grad_in_timing": False,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "optimizer": None if optimizer is None else f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            "baseline_allocated_mib": baseline_bytes / 1024**2,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "peak_reserved_mib": peak_reserved / 1024**2,
            "timings_ms": records,
            "timing_summary_ms": {name: {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
                                  for name, values in records.items()},
            "memory_scope": "Total CUDA allocator memory during measurement, including backward when enabled; not activation-only",
        }
        # 【新增 2026-09-17】自动创建结果目录，并用独占写入再次防止覆盖已有文件。
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("x") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results JSON: {args.results_json}")
    if args.memory_snapshot:
        print(f"Memory snapshot: {args.memory_snapshot}")


if __name__ == "__main__":
    main()
