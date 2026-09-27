import os
import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from cs336_basics.model import BasicsTransformerLM
from cs336_systems.optimizer_state_sharding import OptimizerStateSharding



def setup(rank, world_size, backend="nccl"):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"


    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


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

        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.0,
        )


    elif optimizer_type == "sharded":

        return OptimizerStateSharding(
            model.parameters(),
            torch.optim.AdamW,
            lr=1e-3,
            weight_decay=0.0,
        )


    else:

        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}"
        )



def reset_memory():

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()



def get_memory_gb():

    torch.cuda.synchronize()

    return (
        torch.cuda.max_memory_allocated()
        /
        (1024 ** 3)
    )



def print_memory(label, optimizer_type):

    dist.barrier()


    if dist.get_rank() == 0:

        memory = get_memory_gb()

        print(
            f"{optimizer_type:10s} | "
            f"{label:15s} | "
            f"{memory:.3f} GiB"
        )



def create_batch(device):

    batch_size = 4
    context_length = 512
    vocab_size = 10000


    x = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        device=device,
    )


    y = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        device=device,
    )


    return x, y



def train_step(model, optimizer, device):

    optimizer.zero_grad()


    x, y = create_batch(device)


    logits = model(x)


    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )


    loss.backward()


    optimizer.step()



# =====================================================
# Part (a): memory accounting
# =====================================================

def run_experiment(rank, world_size, optimizer_type):

    setup(
        rank,
        world_size,
        backend="nccl"
    )


    device = torch.device(
        f"cuda:{rank}"
    )


    if rank == 0:

        print("=" * 60)
        print(
            f"Running optimizer: {optimizer_type}"
        )
        print("=" * 60)



    reset_memory()



    model = create_model(device)


    optimizer = create_optimizer(
        model,
        optimizer_type
    )


    dist.barrier()



    # after initialization

    print_memory(
        "after_init",
        optimizer_type
    )



    # forward + backward

    x, y = create_batch(device)


    logits = model(x)


    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )


    loss.backward()


    dist.barrier()



    # before optimizer.step

    print_memory(
        "before_step",
        optimizer_type
    )



    optimizer.step()


    dist.barrier()



    # after optimizer.step

    print_memory(
        "after_step",
        optimizer_type
    )


    optimizer.zero_grad()


    cleanup()





# =====================================================
# Part (b): timing
# =====================================================

def run_timing(rank, world_size, optimizer_type):


    setup(
        rank,
        world_size,
        backend="nccl"
    )


    device = torch.device(
        f"cuda:{rank}"
    )



    if rank == 0:

        print("=" * 60)
        print(
            f"Timing optimizer: {optimizer_type}"
        )
        print("=" * 60)



    model = create_model(device)


    optimizer = create_optimizer(
        model,
        optimizer_type
    )


    dist.barrier()



    # --------------------
    # warmup
    # --------------------

    warmup_steps = 5


    for _ in range(warmup_steps):

        train_step(
            model,
            optimizer,
            device
        )



    torch.cuda.synchronize()

    dist.barrier()



    # --------------------
    # measurement
    # --------------------

    measurement_steps = 10


    torch.cuda.synchronize()

    dist.barrier()


    start = time.perf_counter()



    for _ in range(measurement_steps):

        train_step(
            model,
            optimizer,
            device
        )



    torch.cuda.synchronize()

    dist.barrier()


    end = time.perf_counter()



    avg_ms = (
        (end - start)
        /
        measurement_steps
        *
        1000
    )



    if rank == 0:

        print(
            f"{optimizer_type}: "
            f"{avg_ms:.3f} ms/step"
        )



    cleanup()





def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[
            "adamw",
            "sharded"
        ],
        required=True,
    )


    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "memory",
            "timing"
        ],
        default="memory",
    )


    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
    )


    args = parser.parse_args()



    if args.mode == "memory":

        mp.spawn(
            run_experiment,
            args=(
                args.world_size,
                args.optimizer,
            ),
            nprocs=args.world_size,
            join=True,
        )


    elif args.mode == "timing":

        mp.spawn(
            run_timing,
            args=(
                args.world_size,
                args.optimizer,
            ),
            nprocs=args.world_size,
            join=True,
        )




if __name__ == "__main__":

    main()