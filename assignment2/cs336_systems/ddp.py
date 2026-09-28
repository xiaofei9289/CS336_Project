import torch
from torch import nn
import torch.distributed as dist


class DDPNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        # Broadcast parameters from rank 0 to every rank.
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Average gradients across ranks. Call after backward."""
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


class DDPBatch(DDPNaive):
    def finish_gradient_synchronization(self):
        """Flatten grads, one all-reduce, then unflatten back. Call after backward."""
        world_size = dist.get_world_size()

        gradients = [param.grad for param in self.module.parameters() if param.grad is not None]

        flat_grad = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad /= world_size
        synced_grads = torch._utils._unflatten_dense_tensors(flat_grad, gradients)
        for old_grad, new_grad in zip(gradients, synced_grads):
            old_grad.copy_(new_grad)


class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []

        # Broadcast parameters from rank 0 to every rank.
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        # Launch async all-reduce as each parameter's gradient is ready.
        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._async_all_reduce)

    def _async_all_reduce(self, param):
        """Fired after this parameter's gradient is accumulated."""
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, param))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all async all-reduces, then average. Call after backward."""
        world_size = dist.get_world_size()
        for handle, param in self.handles:
            handle.wait()
            param.grad /= world_size
        self.handles.clear()
