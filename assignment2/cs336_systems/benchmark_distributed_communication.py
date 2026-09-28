import os
import csv
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_WARMUPS = 5
NUM_STEPS = 10


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def all_reduce_benchmark(rank, world_size, data_size, backend):
    # GPU / NCCL setup
    use_gpu = backend == "nccl"
    if use_gpu:
        torch.cuda.set_device(rank)
    setup(rank, world_size, backend)

    # Allocate the tensor; float32 is 4 bytes per element.
    num_elements = (data_size * 1024**2) // 4
    device = f"cuda:{rank}" if use_gpu else "cpu"
    data = torch.randn(num_elements, dtype=torch.float32, device=device)

    # Warmup and timed steps share `data`. All-reduce time does not depend on the values.
    for _ in range(NUM_WARMUPS):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if use_gpu:
            torch.cuda.synchronize()

    # Measurement
    if use_gpu:
        torch.cuda.synchronize()
    dist.barrier()
    if use_gpu:
        torch.cuda.synchronize()

    start = timeit.default_timer()
    for _ in range(NUM_STEPS):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    if use_gpu:
        torch.cuda.synchronize()
    end = timeit.default_timer()

    elapsed_time = (end - start) / NUM_STEPS
    all_times = [None] * world_size
    dist.all_gather_object(all_times, elapsed_time)

    if rank == 0:
        avg_ms = sum(all_times) / len(all_times) * 1000
        max_ms = max(all_times) * 1000
        print("======================")
        print(f"Backend: {backend}")
        print(f"Data size: {data_size} MB")
        print(f"Processes: {world_size}")
        print("Rank times:")
        for i, t in enumerate(all_times):
            print(f"rank {i}: {t*1000:.4f} ms")
        print(f"Average: {avg_ms:.4f} ms")
        print(f"Max: {max_ms:.4f} ms")

        result_file = "all_reduce_results.csv"
        file_exists = os.path.exists(result_file)
        with open(result_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["backend", "world_size", "tensor_MB", "avg_ms", "max_ms"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "backend": backend,
                "world_size": world_size,
                "tensor_MB": data_size,
                "avg_ms": avg_ms,
                "max_ms": max_ms,
            })

    dist.destroy_process_group()


def benchmark(backend):
    # A800 NCCL sweep. Mac smoke tests used data_sizes=[1], num_processes=[2], backend="gloo".
    # Handout settings are 2, 4, or 6. 8 is an extra count when the machine has enough GPUs.
    # Skip any count this machine cannot place so later sizes still run.
    data_sizes = [1, 10, 100, 1024]
    num_processes = [2, 4, 6, 8]
    if backend == "nccl":
        visible_gpus = torch.cuda.device_count()
        skipped = [n for n in num_processes if n > visible_gpus]
        num_processes = [n for n in num_processes if n <= visible_gpus]
        if skipped:
            print(
                f"Visible GPUs: {visible_gpus}. "
                f"Skip process counts {skipped}; remaining sizes still run."
            )
        if not num_processes:
            raise RuntimeError(f"Need at least 2 visible GPUs for NCCL, found {visible_gpus}.")
    for size in data_sizes:
        for world_size in num_processes:
            mp.spawn(
                all_reduce_benchmark,
                args=(world_size, size, backend),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    # A800 timing uses NCCL. Switch to "gloo" for a Mac smoke test.
    benchmark("nccl")
