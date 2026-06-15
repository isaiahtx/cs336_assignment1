import einx
from numpy import require
from torch import Tensor
import torch


def cross_entropy(logits: Tensor, targets: Tensor):
    """
    logits: (... vocab_size)
    targets: (...)
    """

    logits = logits - logits.max(-1, True).values
    x = einx.get_at("... [vocab_size], ... -> ...", logits, targets)
    x = torch.log(einx.sum("... [vocab_size]", torch.exp(logits))) - x

    return x.mean()


if __name__ == "__main__":
    x = torch.tensor(2.0,requires_grad=True)
    y = torch.tensor(3.0,requires_grad=True)
    z = x + y
    w = z * 2
    w.backward()

    print(z.grad_fn.next_functions)
    print(type(z.grad_fn))
    print(type(z.grad_fn).__mro__)
