import torch
from torch import nn, Tensor
import einx
import numpy as np

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        std: float = np.sqrt(2 / (in_features + out_features))
        self.W: nn.Parameter = nn.Parameter(torch.empty(out_features,in_features,dtype=dtype,device=device))
        nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d_in)

        returns: (d_out)
        """
        return einx.dot("d_out [d_in], ... [d_in] -> ... d_out",self.W,x)