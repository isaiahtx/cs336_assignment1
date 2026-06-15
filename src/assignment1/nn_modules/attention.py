import torch
import torch.nn as nn
from torch import Tensor
import einx
import numpy as np

from .rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
    print(Q.shape,K.shape,(mask if mask is not None else torch.tensor([])).shape)
    logits = einx.dot("... q dk, ... k dk -> ... q k",Q,K) / np.sqrt(Q.shape[-1])
    if mask is not None:
        logits[~mask] = -torch.inf

    probs = einx.softmax("... q [k]",logits)

    out = einx.dot("... q [k], ... [k] dv -> ... q dv",probs,V)
    return out


def apply_rope_to_qk(Q: Tensor, rope: RotaryPositionalEmbedding, num_heads: int, token_positions: Tensor | None = None) -> Tensor:
    token_positions = einx.id("... seq -> ... one seq",token_positions,one=1) if token_positions is not None else None
    Q = einx.id("... seq (h dk) -> ... h seq dk",Q,h=num_heads)
    Q= rope.forward(Q,token_positions)
    return einx.id("... h q dk -> ... q (h dk)", Q,h=num_heads)


class CausalMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None
    ):
        super().__init__()

        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.rope = rope

        self.W = nn.Parameter(torch.empty((num_heads * (d_k + d_k + d_v),d_model),dtype=dtype,device=device))
        self.WO = nn.Parameter(torch.empty((d_model,num_heads * d_v),dtype=dtype,device=device))

        for mat in (self.W,self.WO):
            std = np.sqrt(2 / (mat.shape[0] + mat.shape[1]))
            nn.init.trunc_normal_(mat, std=std, a=-3*std, b = 3*std)
    
    def flops(self, seq: int) -> int:
        # number of flops needed for a single foward pass on a sequence of length seq
        mult1 = (2 * seq * self.d_model * self.num_heads * (2 * self.d_k + self.d_v))
        mult2 = 2 * (seq ** 2) * self.d_k * self.num_heads
        mult3 = 2 * (self.num_heads ** 2) * (seq ** 2) * self.d_v
        mult4 = 2 * (seq ** 2) * self.d_v * self.num_heads

        return mult1 + mult2 + mult3 + mult4
    
    def forward(self,x: Tensor, token_positions: Tensor | None = None) -> torch.Tensor:
        """
        x: (... seq d_model)
        token_positions: (... seq) | None

        returns: (... seq d_model)
        """
        h = self.num_heads
        dk = self.d_k
        dv = self.d_v
        seq_len: int | None = None
        if len(x.shape) > 1:
            seq_len = x.shape[-2]

        # Multiplication 1
        Q, K, V = torch.split(einx.dot("stacked [d_model], ... seq [d_model] -> ... seq stacked", self.W, x),[h*dk,h*dk,h*dv],-1)

        if seq_len is not None and self.rope is not None:
            Q = apply_rope_to_qk(Q,self.rope,h,token_positions=token_positions)
            K = apply_rope_to_qk(K,self.rope,h,token_positions=token_positions)

        # Multiplication 2
        logits = einx.dot("... q (h [dk]), ... k (h [dk]) -> ... h q k",Q,K,h=h) / np.sqrt(dk)

        if seq_len is not None:
            mask = torch.triu(torch.ones_like(logits,dtype=torch.bool),diagonal=1)
            logits[mask] = -torch.inf

        probs = einx.softmax("... h q [k]",logits)

        # Multiplication 3
        mha = einx.dot("... h q [k], ... [k] (h dv) -> ... q (h dv)",probs,V,h=h)

        # Multiplication 4
        return einx.dot("d_model [h_dv], ... seq [h_dv] -> ... seq d_model",self.WO,mha,h=h)