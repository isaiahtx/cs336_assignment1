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

class SGD(Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3
        ):
        defaults = {"lr":lr}
        super().__init__(params,defaults)
    
    def step(
        self,
        closure: Callable | None = None
    ) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
            
        return loss


if __name__ == "__main__":

    from tqdm.auto import tqdm

    weights = torch.nn.Parameter(5* torch.randn((10,10)))
    opt = SGD([weights],lr=5)

    print(f"weights before: {weights}")

    it = 0
    for t in tqdm(list(range(1_000_000))):
        it += 1
        opt.zero_grad()
        loss = (weights ** 2).mean()
        #print(f"\tit {t}:",loss.cpu().item())
        loss.backward()
        opt.step()
    
    print(f"weights after: {weights}")