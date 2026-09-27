# cs336_systems/fsdp.py

import torch
from torch import nn
import torch.distributed as dist

from cs336_basics.model import Linear, Embedding


class FullyShardedDataParallel(nn.Module):

    def __init__(
        self,
        module: nn.Module,
        compute_dtype=None
    ):
        super().__init__()

        self.module = module
        self.compute_dtype = compute_dtype

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.sharded_modules = []
        self.replicated_modules = []


        self._setup_parameter_sharding()

        self._setup_prefetch()


        if self.compute_dtype is not None:
            self._register_mixed_precision_hooks()



    # ============================================================
    # IMPORTANT: nn.Module forward
    # ============================================================

    def forward(self, *args, **kwargs):

        return self.module(
            *args,
            **kwargs
        )



    # ============================================================
    # Parameter Sharding
    # ============================================================

    def _setup_parameter_sharding(self):

        for module in self.module.modules():

            if isinstance(module, (Linear, Embedding)):

                self._shard_module(module)

            else:

                if hasattr(module, "weight"):

                    if module.weight is not None:
                        self.replicated_modules.append(module)



    def _shard_module(self, module):

        weight = module.weight.data


        chunks = torch.chunk(
            weight,
            self.world_size,
            dim=0
        )


        local_weight = chunks[self.rank].clone()


        module._fsdp_is_sharded = True
        module._fsdp_is_full = False


        module.weight = nn.Parameter(
            local_weight,
            requires_grad=True
        )


        self.sharded_modules.append(module)



    # ============================================================
    # Prefetch setup
    # ============================================================

    def _setup_prefetch(self):

        for idx, module in enumerate(
            self.sharded_modules
        ):

            module._fsdp_index = idx


            module.register_forward_pre_hook(
                self._make_fsdp_forward_pre_hook()
            )


            module.register_forward_hook(
                self._make_fsdp_forward_post_hook()
            )


            module.register_full_backward_pre_hook(
                self._make_fsdp_backward_pre_hook()
            )



    # ============================================================
    # Gather / Release
    # ============================================================

    def _gather_module(self, module):

        if module._fsdp_is_full:
            return


        shard = module.weight.data


        gathered = [
            torch.zeros_like(shard)
            for _ in range(self.world_size)
        ]


        dist.all_gather(
            gathered,
            shard
        )


        full_weight = torch.cat(
            gathered,
            dim=0
        )


        module._fsdp_saved_shard = (
            shard.clone()
        )


        module.weight.data = full_weight

        module._fsdp_is_full = True



    def _release_module(self, module):

        if not module._fsdp_is_full:
            return


        module.weight.data = (
            module._fsdp_saved_shard.clone()
        )


        module._fsdp_is_full = False



    # ============================================================
    # Forward hooks for prefetch
    # ============================================================

    def _make_fsdp_forward_pre_hook(self):

        def hook(module, inputs):

            self._gather_module(module)


        return hook



    def _make_fsdp_forward_post_hook(self):

        def hook(module, inputs, output):

            idx = module._fsdp_index


            # release current layer

            self._release_module(module)


            # prefetch i+2

            next_idx = idx + 2


            if next_idx < len(
                self.sharded_modules
            ):

                self._gather_module(
                    self.sharded_modules[next_idx]
                )


        return hook



    # ============================================================
    # Backward hook
    # ============================================================

    def _make_fsdp_backward_pre_hook(self):

        def hook(module, grad_output):

            self._gather_module(module)


        return hook



    # ============================================================
    # Mixed Precision
    # ============================================================

    def _register_mixed_precision_hooks(self):

        for module in self.module.modules():

            if not isinstance(
                module,
                (Linear, Embedding)
            ):
                continue


            module.register_forward_pre_hook(
                self._make_mp_forward_pre_hook()
            )


            module.register_forward_hook(
                self._make_mp_forward_post_hook()
            )


            if isinstance(module, Linear):

                module.register_full_backward_pre_hook(
                    self._make_mp_backward_pre_hook()
                )


            module.weight.register_post_accumulate_grad_hook(
                self._make_grad_hook()
            )



    def _make_mp_forward_pre_hook(self):

        def hook(module, inputs):

            module._saved_fp32_weight = (
                module.weight.data
            )


            module.weight.data = (
                module.weight.data
                .to(self.compute_dtype)
            )


        return hook



    def _make_mp_forward_post_hook(self):

        def hook(module, inputs, output):

            # FSDP forward post hook 已经负责:
            #
            # full weight -> shard weight
            #
            # mixed precision hook 不应该重新写回 full fp32 weight
            #
            # 否则会导致:
            #
            # weight.shape = full
            # _fsdp_is_full = False
            #
            # backward 时:
            # shard 被错误认为需要 gather
            # full 再 all_gather
            # 产生 2x size


            if hasattr(
                module,
                "_saved_fp32_weight"
            ):
                del module._saved_fp32_weight


        return hook



    def _make_mp_backward_pre_hook(self):

        def hook(module, grad_output):

            # 保存 fp32 master

            module._saved_fp32_weight_bwd = (
                module.weight.data
            )


            # backward compute dtype

            module.weight.data = (
                module.weight.data
                .to(self.compute_dtype)
            )


            module.weight.grad = None


        return hook



    def _make_grad_hook(self):

        def hook(param):

            # backward 使用 fp16 后，
            # 先恢复 fp32 master weight
            #
            # 再把 grad 转 fp32


            module = None


            for m in self.sharded_modules:

                if m.weight is param:
                    module = m
                    break


            if module is not None:


                if hasattr(
                    module,
                    "_saved_fp32_weight_bwd"
                ):

                    module.weight.data = (
                        module._saved_fp32_weight_bwd
                    )

                    del module._saved_fp32_weight_bwd



            if param.grad is not None:

                param.grad = (
                    param.grad.float()
                )


        return hook



    # ============================================================
    # Gradient Synchronization
    # ============================================================

    def finish_gradient_synchronization(self):


        for module in self.sharded_modules:


            param = module.weight


            if param.grad is None:
                continue


            full_grad = param.grad


            chunks = list(
                full_grad.chunk(
                    self.world_size,
                    dim=0
                )
            )


            local_grad = torch.empty_like(
                chunks[self.rank]
            )


            dist.reduce_scatter(
                local_grad,
                chunks,
                op=dist.ReduceOp.SUM
            )


            local_grad /= self.world_size



            self._release_module(module)


            param.grad = local_grad



        for module in self.replicated_modules:


            param = module.weight


            if param.grad is None:
                continue


            dist.all_reduce(
                param.grad
            )


            param.grad /= self.world_size



    # ============================================================
    # Gather Full Params
    # ============================================================

    def gather_full_params(self):

        result = {}


        module_map = dict(
            self.module.named_modules()
        )


        for name, param in self.module.named_parameters():


            module_name = ".".join(
                name.split(".")[:-1]
            )


            module = module_map.get(
                module_name,
                self.module
            )


            if hasattr(
                module,
                "_fsdp_is_sharded"
            ):


                gathered = [
                    torch.zeros_like(param.data)
                    for _ in range(self.world_size)
                ]


                dist.all_gather(
                    gathered,
                    param.data
                )


                result[name] = torch.cat(
                    gathered,
                    dim=0
                )


            else:

                result[name] = param.data.clone()


        return result