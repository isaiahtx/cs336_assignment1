import torch
import numpy as np
import numpy.typing as npt
from torch import Tensor

def data_loading(
    x: npt.NDArray[np.int64],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    idxs = np.random.randint(0, x.shape[0] - context_length, size=batch_size)
    idxs = idxs[:, None] + np.arange(context_length)
    inputs = torch.as_tensor(x[idxs],device=device)
    targets = torch.as_tensor(x[idxs + 1],device=device)
    return inputs, targets