import torch
from torch import nn, Tensor
from .rope import RotaryPositionalEmbedding
from .normalization import RMSNorm
from .attention import CausalMHA
from .activation import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        theta: float = 10_000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads
        
        self.ln1 = RMSNorm(
            d_model=d_model,
            dtype=dtype,
            device=device
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            dtype=dtype,
            device=device
        )
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_k,
            max_seq_len=max_seq_len,
            device=device
        )
        self.cmha = CausalMHA(
            d_model,
            num_heads,
            d_k,
            d_v,
            rope=self.rope,
            dtype=dtype,
            device=device
        )
        self.ffnn = SwiGLU(
            d_model,
            d_ff,
            device=device,
            dtype=dtype
        )
        

    def forward(self, x:Tensor) -> Tensor:
        y1 = self.cmha.forward(self.ln1.forward(x))
        z1 = x + y1
        y2 = self.ffnn(self.ln2.forward(z1))
        z2 = y2 + z1
        return z2