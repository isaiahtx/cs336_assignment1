import torch
import torch.nn as nn
import einx

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim

        self.W: nn.Parameter = nn.Parameter(torch.empty((num_embeddings, embedding_dim),dtype=dtype,device=device))
        nn.init.trunc_normal_(self.W, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return einx.get_at("[vocab_size] d_model, ... -> ... d_model", self.W, token_ids)