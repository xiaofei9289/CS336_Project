import torch
from typing import Type, Any
import torch.distributed as dist


class OptimizerStateSharding(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any
    ):

        self.optimizer_cls = optimizer_cls
        self.optim = None

        # parameter -> owner rank
        self.param_to_rank = {}

        # global parameter counter
        self.param_index = 0

        # initialize outer optimizer
        super().__init__(
            params,
            defaults=kwargs
        )


    def add_param_group(
        self,
        param_group: dict[str, Any]
    ):

        # keep all parameters in outer optimizer
        # so zero_grad() works correctly
        super().add_param_group(param_group)


        rank = dist.get_rank()
        world_size = dist.get_world_size()


        local_params = []


        # round-robin assign parameters
        for param in param_group["params"]:

            owner = self.param_index % world_size

            self.param_to_rank[param] = owner


            if owner == rank:
                local_params.append(param)


            self.param_index += 1



        # this rank owns no parameters in this group
        if len(local_params) == 0:
            return



        # create local optimizer param group
        local_group = param_group.copy()
        local_group["params"] = local_params



        if self.optim is None:

            self.optim = self.optimizer_cls(
                [local_group],
                **self.defaults
            )

        else:

            self.optim.add_param_group(
                local_group
            )



    def step(
        self,
        closure=None,
        **kwargs
    ):

        # only update local parameters
        if self.optim is not None:

            loss = self.optim.step(
                closure,
                **kwargs
            )

        else:

            loss = None



        # synchronize updated parameters
        for group in self.param_groups:

            for param in group["params"]:

                dist.broadcast(
                    param.data,
                    src=self.param_to_rank[param]
                )


        return loss