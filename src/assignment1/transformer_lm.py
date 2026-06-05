import torch
from torch import Tensor, nn
from .nn_modules import TransformerBlock, Embedding, RMSNorm, Linear
from typing import Self
# import einx

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype
        )

        self.transformer_blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=context_length,
                d_ff=d_ff,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ]

        self.ln_final = RMSNorm(d_model,device=device,dtype=dtype)
        self.lm_head = Linear(d_model,vocab_size)

    @classmethod
    def from_weights(
            cls,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            weights: dict[str,Tensor]
        ) -> Self:
        tlm = cls(
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta
        )

        for i in range(num_layers):
            bw: dict[str,Tensor] = {
                name: weights[f"layers.{i}.{name}"]
                for name in [
                    "attn.q_proj.weight",
                    "attn.k_proj.weight",
                    "attn.v_proj.weight",
                    "attn.output_proj.weight",
                    'ln1.weight',
                    'ln2.weight',
                    'ffn.w1.weight',
                    'ffn.w2.weight',
                    'ffn.w3.weight'
                ]
            }
            tlm.transformer_blocks[i] = TransformerBlock.from_weights(
                d_model,
                num_heads,
                context_length,
                d_ff,
                rope_theta,
                bw
            )

        Wemb = weights['token_embeddings.weight']
        Wln_final = weights['ln_final.weight']
        Wlm_head = weights['lm_head.weight']

        tlm.embedding.load_state_dict({"W":Wemb})
        tlm.ln_final.load_state_dict({"gain":Wln_final})
        tlm.lm_head.load_state_dict({"W":Wlm_head})
        
        return tlm
    
    def forward(self, indices: Tensor) -> Tensor:
        x = self.embedding.forward(indices)

        for tb in self.transformer_blocks:
            x = tb.forward(x)
        
        y = self.lm_head.forward(self.ln_final.forward(x))

        return y

        # return einx.softmax("... seq [vocab]",y)