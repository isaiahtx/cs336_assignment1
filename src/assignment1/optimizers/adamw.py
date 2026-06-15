from torch.optim import Optimizer
from collections.abc import Iterable
from typing import TypeAlias, Any, Callable
import math
import torch

ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)

class AdamW(Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            betas: tuple[float,float] = (0.9,0.95),
            eps: float = 1e-8
        ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params,defaults)
    
    def step(
        self,
        closure: Callable | None = None
    ) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1,beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t",1)
                m = state.get("m",torch.zeros_like(p))
                v = state.get("v",torch.zeros_like(p))

                g = p.grad.data
                alpha_t = lr * math.sqrt(1-(beta2**t)) / (1-(beta1 ** t))
                p.data = p.data * (1 - lr * weight_decay)
                m = beta1 * m + (1-beta1) * g
                v = beta2 * v + (1-beta2) * (g ** 2)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
            
        return loss