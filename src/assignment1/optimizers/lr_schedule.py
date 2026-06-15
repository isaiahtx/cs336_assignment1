from torch.optim import Optimizer
from collections.abc import Iterable
from typing import TypeAlias, Any, Callable
import math
import torch

def learning_rate_schedule(
    t: int,
    amax: float,
    amin: float,
    Tw: int,
    Tc: int
) -> float:
    if t < Tw:
        return amax * t / Tw
    if t <= Tc:
        return amin + 0.5 * (1 + math.cos((t - Tw) * math.pi / (Tc - Tw))) * (amax - amin)
    else:
        return amin