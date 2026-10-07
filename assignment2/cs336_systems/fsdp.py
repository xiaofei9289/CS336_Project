"""Layer-wise FSDP for the assignment's Linear and Embedding modules.

The first forward records execution order; later forwards prefetch at most two
calls ahead. Use prefetch=False for data-dependent execution order. Create the
optimizer after wrapping; master parameters and reduced gradients are FP32.
"""

from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from cs336_basics.model import Embedding, Linear


class _LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master, fsdp, module):
        full = fsdp._consume(module, "forward")
        ctx.input_dtype = x.dtype
        x = x.to(full.dtype)
        with fsdp._range("compute", module, "forward"):
            y = F.linear(x, full)
        ctx.save_for_backward(x)
        ctx.fsdp, ctx.module = fsdp, module
        module._full = None
        fsdp._after_forward(module)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        fsdp, module = ctx.fsdp, ctx.module
        (x,) = ctx.saved_tensors
        full = fsdp._consume(module, "backward")
        with fsdp._range("compute", module, "backward"):
            grad_y = grad_y.to(full.dtype)
            grad_x = (grad_y @ full).to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
            grad_w = grad_y.reshape(-1, grad_y.shape[-1]).T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
        module._full = None
        del full
        local = fsdp._reduce_gradient(module, grad_w) if grad_w is not None else None
        return grad_x, local, None, None


class _EmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, master, fsdp, module):
        full = fsdp._consume(module, "forward")
        with fsdp._range("compute", module, "forward"):
            y = F.embedding(token_ids, full)
        ctx.save_for_backward(token_ids)
        ctx.fsdp, ctx.module = fsdp, module
        module._full = None
        fsdp._after_forward(module)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        fsdp, module = ctx.fsdp, ctx.module
        (token_ids,) = ctx.saved_tensors
        # Embedding gradients depend on indices, not on weight values.
        with fsdp._range("compute", module, "backward"):
            grad_w = torch.zeros(module._full_shape, dtype=torch.float32, device=grad_y.device)
            grad_w.index_add_(0, token_ids.reshape(-1), grad_y.reshape(-1, grad_y.shape[-1]).float())
        return None, fsdp._reduce_gradient(module, grad_w), None, None


def _linear_forward(module, x):
    return _LinearFn.apply(x, module.weight, module._fsdp, module)


def _embedding_forward(module, token_ids):
    return _EmbeddingFn.apply(token_ids, module.weight, module._fsdp, module)


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype=None, *, prefetch=True, profile=False):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.prefetch = prefetch
        self.profile = profile
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharded_modules = []
        self._pending = {}
        self._prefetch_q = []
        self._forward_order = None
        self._observed_order = []
        self._position = 0
        shared = {}
        for name, child in module.named_modules():
            if not isinstance(child, (Linear, Embedding)):
                continue
            original = child.weight
            if original.dtype != torch.float32:
                raise ValueError("FSDP expects FP32 master parameters before wrapping")
            child._full_shape = tuple(original.shape)
            child._shard_rows = (original.shape[0] + self.world_size - 1) // self.world_size
            if id(original) not in shared:
                rows = child._shard_rows
                padded = F.pad(original.detach(), (0, 0, 0, rows * self.world_size - original.shape[0]))
                local = padded.narrow(0, self.rank * rows, rows).clone()
                shared[id(original)] = nn.Parameter(local, requires_grad=original.requires_grad)
            child.weight = shared[id(original)]
            child._full = None
            child._fsdp_name = name or "root"
            # Avoid registering the wrapper as a child of the wrapped layer.
            object.__setattr__(child, "_fsdp", self)
            child.forward = MethodType(_linear_forward if isinstance(child, Linear) else _embedding_forward, child)
            self.sharded_modules.append(child)
        sharded_ids = {id(child.weight) for child in self.sharded_modules}
        self.replicated_params = [p for p in module.parameters() if id(p) not in sharded_ids]
        if any(p.dtype != torch.float32 for p in self.replicated_params):
            raise ValueError("FSDP expects FP32 replicated parameters")

    @contextmanager
    def _range(self, operation, module, phase):
        if self.profile and module.weight.is_cuda:
            torch.cuda.nvtx.range_push(f"fsdp:{phase}:{operation}:{module._fsdp_name}")
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield

    def forward(self, *args, **kwargs):
        self._release_gathers()
        self._observed_order = []
        self._position = 0
        try:
            output = self.module(*args, **kwargs)
            if self.prefetch and self._forward_order is not None and self._observed_order != self._forward_order:
                raise RuntimeError("FSDP execution order changed; use prefetch=False for dynamic models")
            self._forward_order = list(self._observed_order)
            return output
        finally:
            self._release_gathers()

    def _after_forward(self, module):
        self._observed_order.append(module)
        if self.prefetch and self._forward_order is not None:
            if self._position >= len(self._forward_order) or self._forward_order[self._position] is not module:
                raise RuntimeError("FSDP execution order changed; use prefetch=False for dynamic models")
            # Position p+2 may start only after p's forward returns.
            target = self._position + 2
            if target < len(self._forward_order):
                self._enqueue(self._forward_order[target])
        self._position += 1

    def _enqueue(self, module):
        if module._full is not None or module in self._pending or module in self._prefetch_q:
            return
        self._prefetch_q.append(module)
        self._pump()

    def _pump(self):
        if self._pending or not self._prefetch_q:
            return
        self._begin_gather(self._prefetch_q.pop(0), "forward")

    def _begin_gather(self, module, phase):
        if module in self._pending or module._full is not None:
            return
        shard = module.weight.detach().to(self.compute_dtype or torch.float32).contiguous()
        buffers = [torch.empty_like(shard) for _ in range(self.world_size)]
        with self._range("all_gather", module, phase):
            work = dist.all_gather(buffers, shard, async_op=True)
        self._pending[module] = (work, buffers, shard)

    def _finish_pending(self, module, phase):
        work, buffers, shard = self._pending.pop(module)
        with self._range("wait_weights", module, phase):
            work.wait()
            module._full = torch.cat(buffers, dim=0)[:module._full_shape[0]]

    def _consume(self, module, phase):
        if module._full is None:
            if module not in self._pending and self._pending:
                self._finish_pending(next(iter(self._pending)), phase)
            self._begin_gather(module, phase)
            self._finish_pending(module, phase)
        if phase == "forward" and self.prefetch and self._forward_order and self._position == 0 and len(self._forward_order) > 1:
            # The second call has no layer two before it, so it may overlap the first call.
            self._enqueue(self._forward_order[1])
        if phase == "forward":
            self._pump()
        return module._full

    def _release_gathers(self):
        for work, buffers, shard in list(self._pending.values()):
            work.wait()
        self._pending.clear()
        self._prefetch_q.clear()
        for module in self.sharded_modules:
            module._full = None

    def _reduce_gradient(self, module, grad_w):
        grad_w = grad_w.float()
        rows = module._shard_rows
        padded = F.pad(grad_w, (0, 0, 0, rows * self.world_size - grad_w.shape[0]))
        chunks = list(padded.split(rows, dim=0))
        local = torch.empty_like(module.weight)
        with self._range("reduce_scatter", module, "backward"):
            work = dist.reduce_scatter(local, chunks, op=dist.ReduceOp.SUM, async_op=True)
            # Autograd must receive a ready tensor. Full-gradient storage is
            # released with this call rather than retained until the step ends.
            work.wait()
            local.div_(self.world_size)
        return local

    def finish_gradient_synchronization(self):
        self._release_gathers()
        # Sharded gradients are averaged before autograd accumulation.
        for param in self.replicated_params:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, async_op=True).wait()
            param.grad.div_(self.world_size)

    @torch.no_grad()
    def gather_full_params(self):
        self._release_gathers()
        by_param = {}
        for module in self.sharded_modules:
            if id(module.weight) not in by_param:
                shard = module.weight.detach().contiguous()
                buffers = [torch.empty_like(shard) for _ in range(self.world_size)]
                dist.all_gather(buffers, shard)
                by_param[id(module.weight)] = torch.cat(buffers, dim=0)[:module._full_shape[0]]
        return {name: by_param[id(param)] if id(param) in by_param else param.detach().clone()
                for name, param in self.module.named_parameters()}
