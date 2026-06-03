import torch
import torch.nn as nn
import einx

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps

        self.gain = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einx.mean("... [d_model]",x ** 2) + self.eps)
        x_normed = einx.divide("... d_model, ...", x, rms)
        return einx.multiply("... d_model, d_model", x_normed, self.gain).to(in_dtype)