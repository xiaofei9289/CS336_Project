import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.fsdp import FullyShardedDataParallel
# ============================================================
# Distributed setup
# ============================================================
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
def cleanup():
    dist.destroy_process_group()
# ============================================================
# Memory helper
# ============================================================
def print_peak_memory(rank, name):
    dist.barrier()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    if rank == 0:
        print(f"{name}: {peak_memory:.2f} GiB")
# ============================================================
# Model
# ============================================================
def build_model(device):
    model = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=2560, num_layers=32, num_heads=32, d_ff=10240, rope_theta=10000)
    model.to(device)
    return model
# ============================================================
# Training step
# ============================================================
def run(rank, world_size, profile):
    setup(rank, world_size)
    # IMPORTANT:
    # reset only once
    # keep the same peak-memory accounting
    # as Chapter 6 benchmark
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(f"cuda:{rank}")
    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = build_model(device)
    fsdp_model = FullyShardedDataParallel(model, profile=profile)
    fsdp_model.train()
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)
    # --------------------------------------------------------
    # after_init
    # --------------------------------------------------------
    print_peak_memory(rank, "after_init")
    # --------------------------------------------------------
    # Fake batch
    # --------------------------------------------------------
    batch_size = 4
    context_length = 512
    input_ids = torch.randint(0, 10000, (batch_size, context_length), device=device)
    targets = torch.randint(0, 10000, (batch_size, context_length), device=device)
    def one_step(step_name):
        optimizer.zero_grad()
        torch.cuda.nvtx.range_push("fsdp_forward")
        logits = fsdp_model(input_ids)
        torch.cuda.nvtx.range_pop()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10000), targets.reshape(-1))
        torch.cuda.nvtx.range_push("fsdp_backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        fsdp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        prefix = f"{step_name}_" if step_name else ""
        print_peak_memory(rank, f"{prefix}before_step")
        optimizer.step()
        torch.cuda.synchronize()
        print_peak_memory(rank, f"{prefix}after_step")
        if rank == 0:
            print(f"{step_name or 'measure'} loss:", loss.item())
    if profile:
        one_step("warmup")
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.nvtx.range_push("measurement")
        one_step("measure")
        torch.cuda.nvtx.range_pop()
    else:
        one_step("")
    cleanup()
# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    world_size = 2
    mp.spawn(run, args=(world_size, args.profile), nprocs=world_size, join=True)
if __name__ == "__main__":
    main()
