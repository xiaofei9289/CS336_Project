import torch
import torch.distributed as dist
from typing import Any, Type


class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optim = None
        self.param_to_rank = {}
        self.param_index = 0
        super().__init__(params, defaults=kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_params = []
        for param in param_group["params"]:
            owner = self.param_index % world_size
            self.param_to_rank[param] = owner
            if owner == rank:
                local_params.append(param)
            self.param_index += 1
        if not local_params:
            return
        local_group = {**param_group, "params": local_params}
        if self.optim is None:
            self.optim = self.optimizer_cls([local_group], **self.defaults)
        else:
            self.optim.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        loss = self.optim.step(closure, **kwargs) if self.optim is not None else None
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    dist.broadcast(param.data, src=self.param_to_rank[param])
        return loss
