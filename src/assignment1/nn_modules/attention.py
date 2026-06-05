import torch
import torch.nn as nn
from torch import Tensor
import einx
import numpy as np

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
    assert False
    return out

class CausalMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
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

        self.W = nn.Parameter(torch.empty((num_heads * (d_k + d_k + d_v),d_model),dtype=dtype,device=device))
        self.WO = nn.Parameter(torch.empty((d_model,num_heads * d_v),dtype=dtype,device=device))

        for mat in (self.W,self.WO):
            std = np.sqrt(2 / (mat.shape[0] + mat.shape[1]))
            nn.init.trunc_normal_(mat, std=std, a=-3*std, b = 3*std)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        h = self.num_heads
        dk = self.d_k
        dv = self.d_v
        q: int | None = None
        if len(x.shape) > 1:
            q = x.shape[-2]

        Q, K, V = torch.split(einx.dot("stacked [d_model], ... q [d_model] -> ... q stacked", self.W, x),[h*dk,h*dk,h*dv],-1)

        logits = einx.dot("... q (h [dk]), ... k (h [dk]) -> ... h q k",Q,K,h=h) / np.sqrt(dk)

        if q is not None:
            mask = torch.triu(torch.ones_like(logits,dtype=torch.bool),diagonal=1)
            logits[mask] = -torch.inf

        probs = einx.softmax("... h q [k]",logits)

        mha = einx.dot("... h q [k], ... [k] (h dv) -> ... q (h dv)",probs,V,h=h)

        return einx.dot("dm [(h dv)], ... q [(h dv)] -> ... q dm",self.WO,mha,h=h)