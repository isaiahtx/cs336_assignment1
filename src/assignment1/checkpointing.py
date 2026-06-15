from pathlib import Path
from typing import BinaryIO, IO
from os import PathLike
import torch.nn as nn
from torch.optim import Optimizer
import torch

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | Path | PathLike[str] | BinaryIO | IO[bytes]
) -> None:
    obj = (
        model.state_dict(),
        optimizer.state_dict(),
        iteration,
    )
    torch.save(obj,out)


def load_checkpoint(
    src: str | Path | PathLike[str] | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer
) -> int:
    model_state, optim_state, iteration = torch.load(src)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    return iteration