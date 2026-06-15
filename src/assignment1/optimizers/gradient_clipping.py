from collections.abc import Iterable
import torch
import torch.nn as nn

def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    with torch.no_grad():
        params = [p for p in parameters if p.grad is not None]
        norm = sum((p.grad**2).sum() for p in params).sqrt()
        if norm > max_l2_norm:
            scale = max_l2_norm / (norm + eps)
            for p in params:
                p.grad.mul_(scale)
