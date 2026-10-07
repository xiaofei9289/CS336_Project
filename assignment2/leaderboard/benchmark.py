"""Time one leaderboard training step, or check it against BasicsTransformerLM.

CPU check, aliased onto one device:

    python -m leaderboard.benchmark --check

4 x RTX PRO 6000, handout configuration, PyTorch scaled_dot_product_attention.
Clear caches first. warmup and rep are milliseconds:

    rm -rf ~/.triton/cache
    mkdir -p /tmp/empty-inductor
    TORCHINDUCTOR_CACHE_DIR=/tmp/empty-inductor python -m leaderboard.benchmark

Add --once to run a single step before the timed loop. --layers and
--context-length shrink the model while keeping the same train_step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cs336-basics"))
sys.path.insert(0, str(ROOT))

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from leaderboard.model import LeaderboardAdamW, LeaderboardLM


HANDOUT = dict(
    vocab_size=151936,
    context_length=32768,
    d_model=4096,
    d_ff=11008,
    num_layers=34,
    num_heads=32,
    batch_size=2,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare a tiny model with BasicsTransformerLM")
    parser.add_argument("--once", action="store_true", help="run one training step and exit")
    parser.add_argument("--breakdown", action="store_true", help="time forward, backward, attention, loss, and AdamW")
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument("--rep-ms", type=int, default=30_000)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true", help="torch.compile each transformer block")
    return parser.parse_args()


def handout_devices():
    count = torch.cuda.device_count()
    if count >= 4:
        width = 4
    elif count >= 2:
        width = 2
    else:
        raise SystemExit(f"the handout run needs 2 or 4 visible CUDA devices, found {count}")
    return [torch.device(f"cuda:{index}") for index in range(width)]


def check_against_basics():
    """Shard a tiny staff model and compare logits plus one backward."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    devices = [device, device, device, device]
    config = dict(vocab_size=128, context_length=32, d_model=64, num_layers=2, num_heads=4, d_ff=128)
    reference = BasicsTransformerLM(**config)
    model = LeaderboardLM(
        **config,
        devices=devices,
        dtype=torch.float32,
        attention="pytorch",
        checkpoint_activations=True,
    )
    model.copy_from_basics(reference)
    tokens = torch.randint(0, config["vocab_size"], (2, config["context_length"]))
    targets = torch.randint(0, config["vocab_size"], (2, config["context_length"]))

    reference_logits = reference(tokens)
    with torch.no_grad():
        logits = model(tokens)
    logit_error = (reference_logits - logits).abs().max().item()

    reference_loss = cross_entropy(reference_logits, targets).sum()
    loss = model.nll(tokens, targets)
    reference_loss.backward()
    loss.backward()
    model.sync_dp_grads()

    loss_error = abs(reference_loss.item() - loss.item())
    embed_error = (reference.token_embeddings.weight.grad - model.replica0.embed.grad).abs().max().item()
    q_weight = reference.layers[0].attn.q_proj.weight.grad
    q0, q1 = q_weight.chunk(2, dim=0)
    q_error = max(
        (q0 - model.replica0.blocks[0].attn.q.weight0.grad).abs().max().item(),
        (q1 - model.replica0.blocks[0].attn.q.weight1.grad).abs().max().item(),
    )
    head_weight = reference.lm_head.weight.grad
    h0, h1 = head_weight.chunk(2, dim=0)
    head_error = max(
        (h0 - model.replica0.head.weight0.grad).abs().max().item(),
        (h1 - model.replica0.head.weight1.grad).abs().max().item(),
    )
    print(
        f"logit_max_abs={logit_error:.3e} loss_abs={loss_error:.3e} "
        f"embed_grad={embed_error:.3e} q_grad={q_error:.3e} lm_head_grad={head_error:.3e}"
    )
    limit = 1e-4
    if max(logit_error, loss_error, embed_error, q_error, head_error) > limit:
        raise SystemExit(f"check failed: error exceeded {limit}")
    print("check passed")


def build_timing_model(args):
    config = dict(HANDOUT)
    overrides = {
        "num_layers": args.layers,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    batch_size = config.pop("batch_size")
    devices = handout_devices()
    torch.manual_seed(args.seed)
    model = LeaderboardLM(
        **config,
        devices=devices,
        dtype=torch.bfloat16,
        attention="sdpa",
        checkpoint_activations=not args.no_checkpoint,
        compile_blocks=args.compile,
    )
    return model, devices[0], batch_size, config["context_length"], config["vocab_size"]


def main():
    args = parse_args()
    if args.check:
        check_against_basics()
        return

    model, device, batch_size, context_length, vocab_size = build_timing_model(args)
    labels, targets = torch.randint(0, vocab_size, (2, batch_size, context_length), device=device)
    optimizer = LeaderboardAdamW(model.parameters(), model.dp_pairs)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        model.nll_backward(labels, targets)
        optimizer.step()
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    if args.once or args.breakdown:
        import time

        from leaderboard.model import profile_report, set_profile

        train_step()
        if args.once and not args.breakdown:
            torch.cuda.synchronize()
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
            start = time.perf_counter()
            train_step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peaks = [torch.cuda.max_memory_allocated(index) / 1024**3 for index in range(torch.cuda.device_count())]
            peak_text = " ".join(f"{peak:.1f}" for peak in peaks)
            print(f"step_s={elapsed:.3f} peak_gib={peak_text}")
            return

        set_profile(True)
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        model.nll_backward(labels, targets)
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        adam_start = time.perf_counter()
        optimizer.step()
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        adam_s = time.perf_counter() - adam_start
        step_s = time.perf_counter() - start
        phases = model.phase_s
        print(f"step_s={step_s:.3f}")
        print(f"forward_s={phases['forward']:.3f}")
        print(f"backward_s={phases['backward']:.3f}")
        print(f"adamw_s={adam_s:.3f}")
        print(profile_report())
        return

    import triton

    timing_ms = triton.testing.do_bench(train_step, warmup=args.warmup_ms, rep=args.rep_ms)
    print(timing_ms)


if __name__ == "__main__":
    main()
