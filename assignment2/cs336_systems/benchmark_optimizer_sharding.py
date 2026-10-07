import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.optimizer_state_sharding import OptimizerStateSharding


def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def create_model(device):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10240,
        rope_theta=10000,
    )
    model.to(device)
    return model


def create_optimizer(model, optimizer_type):
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters())
    if optimizer_type == "sharded":
        return OptimizerStateSharding(model.parameters(), torch.optim.AdamW)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def local_batch(input_data, rank, world_size, device):
    local = input_data.shape[0] // world_size
    start = rank * local
    return input_data[start:start + local].to(device)


def forward_backward(model, optimizer, input_data):
    optimizer.zero_grad()
    model(input_data).mean().backward()


def tensor_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def optimizer_state_bytes(optimizer):
    if optimizer is None:
        return 0
    inner = optimizer.optim if isinstance(optimizer, OptimizerStateSharding) else optimizer
    if inner is None:
        return 0
    total = 0
    for state in inner.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def report(label, optimizer_type, model, optimizer):
    torch.cuda.synchronize()
    gib = 1024 ** 3
    payload = {
        "rank": dist.get_rank(),
        "peak": torch.cuda.max_memory_allocated() / gib,
        "allocated": torch.cuda.memory_allocated() / gib,
        "params": tensor_bytes(model.parameters()) / gib,
        "grads": tensor_bytes(p.grad for p in model.parameters() if p.grad is not None) / gib,
        "state": optimizer_state_bytes(optimizer) / gib,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() == 0:
        for row in gathered:
            print(
                f"rank {row['rank']} | {optimizer_type:7s} | {label:11s} | "
                f"peak {row['peak']:.3f} GiB | allocated {row['allocated']:.3f} GiB | "
                f"params {row['params']:.3f} | grads {row['grads']:.3f} | adam {row['state']:.3f}"
            )
    torch.cuda.reset_peak_memory_stats()


def run_experiment(rank, world_size, optimizer_type, input_data):
    setup(rank, world_size, backend="nccl")
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        local = input_data.shape[0] // world_size
        print(f"memory | {optimizer_type} | 1 node x {world_size} GPUs | xl | context 512 | global batch {input_data.shape[0]} | local batch {local}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_model(device)
    report("after_init", optimizer_type, model, None)
    input_data = local_batch(input_data, rank, world_size, device)
    optimizer = create_optimizer(model, optimizer_type)
    forward_backward(model, optimizer, input_data)
    report("before_step", optimizer_type, model, optimizer)
    optimizer.step()
    report("after_step", optimizer_type, model, optimizer)
    cleanup()


def run_timing(rank, world_size, optimizer_type, input_data):
    setup(rank, world_size, backend="nccl")
    device = torch.device(f"cuda:{rank}")
    input_data = local_batch(input_data, rank, world_size, device)
    model = create_model(device)
    optimizer = create_optimizer(model, optimizer_type)
    warmup_steps, measurement_steps = 5, 10
    for _ in range(warmup_steps):
        forward_backward(model, optimizer, input_data)
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(measurement_steps):
        forward_backward(model, optimizer, input_data)
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()
    avg_ms = (time.perf_counter() - start) / measurement_steps * 1000
    if rank == 0:
        print(f"timing | {optimizer_type} | 1 node x {world_size} GPUs | xl | local batch {input_data.shape[0]} | {avg_ms:.3f} ms/step")
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sharded"], required=True)
    parser.add_argument("--mode", type=str, choices=["memory", "timing"], default="memory")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size % args.world_size != 0:
        raise ValueError(f"batch size {args.batch_size} is not divisible by world size {args.world_size}")
    input_data = torch.randint(0, 10000, (args.batch_size, 512))
    worker = run_experiment if args.mode == "memory" else run_timing
    mp.spawn(worker, args=(args.world_size, args.optimizer, input_data), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
