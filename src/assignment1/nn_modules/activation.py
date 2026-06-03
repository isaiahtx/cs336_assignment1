import torch
import torch.nn as nn
import einx
import numpy as np

class SiLU(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 64 * round((8 * d_model / 3) / 64)
            # print(f"Set d_ff = {d_ff} (rounded from {8 * d_model / 3})")

        self.d_ff = d_ff
        self.d_model = d_model

        self.W1 = nn.Parameter(torch.empty((d_ff,d_model),device=device,dtype=dtype))
        self.W2 = nn.Parameter(torch.empty((d_model,d_ff),device=device,dtype=dtype))
        self.W3 = nn.Parameter(torch.empty((d_ff,d_model),device=device,dtype=dtype))
        self.silu = SiLU()

        std = np.sqrt(2 / (d_model + d_ff))
        nn.init.trunc_normal_(self.W1, std=std, a=-3*std, b=std)
        nn.init.trunc_normal_(self.W2, std=std, a=-3*std, b=std)
        nn.init.trunc_normal_(self.W3, std=std, a=-3*std, b=std)
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        L = self.silu(einx.dot("d_ff [d_model], ... [d_model] -> ... d_ff", self.W1, in_features))
        R = einx.dot("d_ff [d_model], ... [d_model] -> ... d_ff", self.W3, in_features)
        return einx.dot("d_model [d_ff], ... [d_ff] -> ... d_model", self.W2, L * R)