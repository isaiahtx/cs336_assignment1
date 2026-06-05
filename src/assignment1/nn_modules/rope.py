import torch
from torch import nn, Tensor
from jaxtyping import Shaped
import einx

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        d2 = d_k // 2

        angles: Tensor = einx.divide(
            "max_seq_len, d2 -> max_seq_len d2",
            torch.arange(0,max_seq_len,device=device),
            theta ** (2 * torch.arange(0,d2,device=device) / d_k)
        )

        cosines: Tensor = einx.id(  # pyright: ignore[reportAssignmentType]
            "max_seq_len d2 -> max_seq_len (d2 two)",
            torch.cos(angles),
            two=2
        )
        alt_sines: Tensor = einx.multiply(
            "max_seq_len d2, two -> max_seq_len (d2 two)",
            torch.sin(angles),
            torch.tensor([-1,1],device=device)
        )

        self.register_buffer('cosines',cosines, persistent=False)
        self.register_buffer('alt_sines', alt_sines, persistent=False)


    def forward(self, x: Shaped[Tensor,"... seq_len d_k"], token_positions: Shaped[Tensor,"... seq_len"] | None = None) -> Shaped[Tensor,"... seq_len d_k"]:
        """
        x: (... seq_len d_k)
        token_positions: (... seq_len) | None

        returns (... seq_len d_k)
        """
        if token_positions is None:
            token_positions = torch.arange(end=x.shape[-2],device=x.device).expand(x.shape[:-1])

        cosines: Tensor = einx.get_at(  # pyright: ignore[reportAssignmentType]
            "[max_seq_len] d_k, ... seq_len -> ... seq_len d_k",
            self.cosines,
            token_positions
        )

        alt_sines: Tensor = einx.get_at(  # pyright: ignore[reportAssignmentType]
            "[max_seq_len] d_k, ... seq_len -> ... seq_len d_k",
            self.alt_sines,
            token_positions
        )

        x_flipped_pairs: Tensor = einx.id(  # pyright: ignore[reportAssignmentType]
            "... d2 two -> ... (d2 two)",
            einx.flip(
                "... d2 [two]",
                einx.id(
                    "... (d2 two) -> ... d2 two",
                    x,
                    two=2
                )
            )
        )
        return cosines * x + alt_sines * x_flipped_pairs

if __name__ == "__main__":
    theta = 10_000
    batch_size = 1
    d_k = 4
    max_seq_len = 2

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len
    )

    x = torch.randn((batch_size,max_seq_len,d_k))
    token_positions = torch.randint(max_seq_len,(batch_size,max_seq_len))

    print(rope(x,token_positions))
