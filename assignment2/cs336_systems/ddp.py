import torch
from torch import nn
import torch.distributed as dist


class DDPNaive(nn.Module):

    def __init__(self, module):

        super().__init__()

        self.module = module


        # ==============================
        # 1. 同步 parameters
        # rank 0 -> all ranks
        # ==============================

        for param in module.parameters():

            dist.broadcast(
                param.data,
                src=0
            )


        # ==============================
        # 2. 同步 buffers
        # 例如:
        # BatchNorm running_mean
        # BatchNorm running_var
        # ==============================

        for buffer in module.buffers():

            dist.broadcast(
                buffer.data,
                src=0
            )



    def forward(self, *inputs, **kwargs):

        return self.module(
            *inputs,
            **kwargs
        )



    def finish_gradient_synchronization(self):

        """
        backward之后调用

        将所有rank的gradient求平均
        """

        world_size = dist.get_world_size()


        for param in self.module.parameters():

            if param.grad is None:
                continue


            # gradient sum

            dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM
            )


            # average gradient

            param.grad /= world_size


class DDPBatch(nn.Module):

    def __init__(self, module):

        super().__init__()

        self.module = module


        # =================================
        # 参数同步
        # rank 0 -> all ranks
        # =================================

        for param in module.parameters():

            dist.broadcast(
                param.data,
                src=0
            )


        # =================================
        # buffer同步
        # 例如 BatchNorm running stats
        # =================================

        for buffer in module.buffers():

            dist.broadcast(
                buffer.data,
                src=0
            )



    def forward(self, *inputs, **kwargs):

        return self.module(
            *inputs,
            **kwargs
        )



    def finish_gradient_synchronization(self):

        """
        backward之后调用

        流程：

        param.grad
             |
             |
        flatten
             |
             |
        all_reduce(sum)
             |
             |
        divide world_size
             |
             |
        unflatten
             |
             |
        写回原grad
        """


        world_size = dist.get_world_size()



        # =================================
        # 收集当前rank的gradient
        # =================================

        gradients = []


        for param in self.module.parameters():

            if param.grad is not None:

                gradients.append(
                    param.grad
                )



        # =================================
        # 判断所有rank是否都有gradient
        #
        # 防止：
        # rank0 没grad直接return
        # rank1 进入all_reduce
        #
        # 导致死锁
        # =================================


        has_grad = torch.tensor(
            int(len(gradients) > 0),
            dtype=torch.int,
            device=next(self.module.parameters()).device
        )


        dist.all_reduce(
            has_grad,
            op=dist.ReduceOp.MIN
        )


        # 所有rank都没有gradient

        if has_grad.item() == 0:

            return



        # =================================
        # flatten gradients
        # =================================

        flat_grad = torch._utils._flatten_dense_tensors(
            gradients
        )



        # =================================
        # 一次all_reduce
        # =================================

        dist.all_reduce(
            flat_grad,
            op=dist.ReduceOp.SUM
        )



        # =================================
        # 求平均gradient
        # =================================

        flat_grad /= world_size



        # =================================
        # unflatten
        # =================================

        synced_grads = torch._utils._unflatten_dense_tensors(
            flat_grad,
            gradients
        )



        # =================================
        # 写回param.grad
        # =================================

        for old_grad, new_grad in zip(
            gradients,
            synced_grads
        ):

            old_grad.copy_(
                new_grad
            )


class DDPOverlapIndividualParameters(nn.Module):

    def __init__(self, module):

        super().__init__()

        self.module = module

        self.handles = []


        # ==========================
        # 参数同步
        # rank0 -> all ranks
        # ==========================

        for param in module.parameters():

            dist.broadcast(
                param.data,
                src=0
            )


        # ==========================
        # buffer同步
        # ==========================

        for buffer in module.buffers():

            dist.broadcast(
                buffer.data,
                src=0
            )


        # ==========================
        # 注册gradient hook
        # ==========================

        for param in module.parameters():

            if param.requires_grad:

                param.register_post_accumulate_grad_hook(
                    self._async_all_reduce
                )



    def _async_all_reduce(self, param):

        """
        当某个parameter梯度计算完成后触发
        """

        handle = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True
        )


        self.handles.append(
            (handle, param)
        )



    def forward(self, *inputs, **kwargs):

        return self.module(
            *inputs,
            **kwargs
        )



    def finish_gradient_synchronization(self):

        """
        backward结束后调用

        等待所有async communication完成
        """

        world_size = dist.get_world_size()


        for handle, param in self.handles:

            handle.wait()

            param.grad /= world_size


        self.handles.clear()
