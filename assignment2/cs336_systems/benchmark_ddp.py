import os
import timeit
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.ddp import DDPBatch, DDPNaive, DDPOverlapIndividualParameters


NUM_WARMUPS = 5
NUM_STEPS = 10


def setup(rank, world_size, backend):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )



def benchmark_ddp(
        rank,
        world_size,
        model_params,
        input_data,
        backend,
        mode
):

    # ==========================
    # NCCL 必须提前绑定GPU
    # ==========================

    use_gpu = (
        torch.cuda.is_available()
        and backend == "nccl"
    )


    if use_gpu:

        torch.cuda.set_device(rank)

        device = torch.device(
            f"cuda:{rank}"
        )

    else:

        device = torch.device("cpu")


    # 初始化通信
    setup(
        rank,
        world_size,
        backend
    )


    # ==========================
    # Model
    # ==========================

    model = BasicsTransformerLM(
        **model_params
    )

    model.to(device)


    if mode == "batch_ddp":

        ddpmodel = DDPBatch(model)

    elif mode == "overlap_params":

        ddpmodel = DDPOverlapIndividualParameters(model)

    elif mode == "naive":

        ddpmodel = DDPNaive(model)

    else:

        raise ValueError(mode)



    # ==========================
    # 每个rank拿自己的batch
    # ==========================

    local_batch = (
        input_data.shape[0]
        //
        world_size
    )


    start = rank * local_batch
    end = start + local_batch


    input_data = input_data[start:end]


    if use_gpu:

        input_data = input_data.to(device)



    optimizer = torch.optim.AdamW(
        ddpmodel.parameters()
    )



    # ==========================
    # Warmup
    # ==========================

    for _ in range(NUM_WARMUPS):

        optimizer.zero_grad(
            set_to_none=True
        )


        output = ddpmodel(
            input_data
        ).mean()


        output.backward()


        ddpmodel.finish_gradient_synchronization()


        optimizer.step()


        if use_gpu:
            torch.cuda.synchronize()



    # ==========================
    # Benchmark
    # ==========================


    dist.barrier()

    if use_gpu:
        torch.cuda.synchronize()



    start_time = timeit.default_timer()


    total_sync_time = 0.0



    for _ in range(NUM_STEPS):


        optimizer.zero_grad(
            set_to_none=True
        )


        output = ddpmodel(
            input_data
        ).mean()


        output.backward()



        # --------------------------------
        # Naive / Flatten:
        # backward之后才通信
        #
        # Overlap:
        # 通信已经嵌入backward
        # 不单独统计
        # --------------------------------

        if mode != "overlap_params":


            sync_start = timeit.default_timer()


            ddpmodel.finish_gradient_synchronization()


            if use_gpu:
                torch.cuda.synchronize()


            sync_end = timeit.default_timer()


            total_sync_time += (
                sync_end - sync_start
            )


        else:

            # overlap:
            #这里只负责wait通信结束

            ddpmodel.finish_gradient_synchronization()



        optimizer.step()



    if use_gpu:

        torch.cuda.synchronize()


    end_time = timeit.default_timer()



    total_time = (
        end_time - start_time
    )


    avg_step_time = (
        total_time
        /
        NUM_STEPS
    )



    if rank == 0:

        print("========================")

        print(
            f"mode: {mode}"
        )


        print(
            f"total {NUM_STEPS} steps: "
            f"{total_time:.6f}s"
        )


        print(
            f"average step: "
            f"{avg_step_time*1000:.4f} ms"
        )


        if mode != "overlap_params":

            print(
                f"average grad sync: "
                f"{total_sync_time/NUM_STEPS*1000:.4f} ms"
            )

        else:

            print(
                "overlap mode: "
                "communication overlaps with backward"
            )



    dist.destroy_process_group()




def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--world_size",
        type=int,
        default=2
    )


    parser.add_argument(
        "--batch_sz",
        type=int,
        default=4
    )


    parser.add_argument(
        "--backend",
        type=str,
        default="gloo"
    )


    parser.add_argument(
        "--tiny",
        action="store_true"
    )


    parser.add_argument(
        "--mode",
        choices=[
            "batch_ddp",
            "overlap_params",
            "naive"
        ],
        default="naive"
    )


    args = parser.parse_args()



    assert (
        args.batch_sz
        %
        args.world_size
        ==
        0
    )



    # ==========================
    # Mac smoke test
    # ==========================

    if args.tiny:

        model_params = {

            "vocab_size":1000,

            "context_length":32,

            "d_model":128,

            "num_layers":2,

            "num_heads":4,

            "d_ff":512,

            "rope_theta":10000,

        }


        context_length = 32



    # ==========================
    # A800正式实验
    # ==========================

    else:

        model_params = {

            "vocab_size":10000,

            "context_length":512,

            "d_model":2560,

            "num_layers":32,

            "num_heads":32,

            "d_ff":10240,

            "rope_theta":10000,

        }


        context_length = 512




    input_data = torch.randint(
        low=0,
        high=model_params["vocab_size"],
        size=(
            args.batch_sz,
            context_length
        )
    )



    mp.spawn(
        benchmark_ddp,
        args=(
            args.world_size,
            model_params,
            input_data,
            args.backend,
            args.mode
        ),
        nprocs=args.world_size,
        join=True
    )



if __name__ == "__main__":

    main()