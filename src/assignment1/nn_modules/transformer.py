import torch
from torch import nn, Tensor
from .rope import RotaryPositionalEmbedding
from .normalization import RMSNorm
from .attention import CausalMHA
from .activation import SwiGLU
from typing import Self

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        theta: float | None = 10_000,
        rope: RotaryPositionalEmbedding | None = None,
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

        if rope is not None:
            self.rope = rope
        elif theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            raise ValueError("One of `rope` or `theta` must be defined")

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


    def flops(self, seq: int) -> int:
        return self.cmha.flops(seq) + seq * self.ffnn.flops()


    @classmethod
    def from_weights(
            cls,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            d_ff: int,
            theta: float,
            weights: dict[str,Tensor],
            d_k: int | None = None,
            d_v: int | None = None,
        ) -> Self:
        WQ = weights["attn.q_proj.weight"]
        WK = weights["attn.k_proj.weight"]
        WV = weights["attn.v_proj.weight"]
        WO = weights["attn.output_proj.weight"]
        Wln1 = weights['ln1.weight']
        Wln2 = weights['ln2.weight']
        W1 = weights['ffn.w1.weight']
        W2 = weights['ffn.w2.weight']
        W3 = weights['ffn.w3.weight']

        tfb = cls(
            d_model,
            num_heads,
            max_seq_len,
            d_ff,
            theta=theta,
            d_k=d_k,
            d_v=d_v
        )

        W = torch.concat((WQ,WK,WV),dim=-2)
        tfb.cmha.load_state_dict({"W":W,"WO":WO})

        tfb.ln1.load_state_dict({"gain":Wln1})
        tfb.ln2.load_state_dict({"gain":Wln2})

        tfb.ffnn.load_state_dict({"W1":W1,"W2":W2,"W3":W3})

        return tfb
        

    def forward(self, x:Tensor) -> Tensor:
        y1 = self.cmha.forward(self.ln1.forward(x))
        z1 = x + y1
        y2 = self.ffnn(self.ln2.forward(z1))
        z2 = y2 + z1
        return z2