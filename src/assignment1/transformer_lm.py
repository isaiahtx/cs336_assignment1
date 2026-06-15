import torch
from torch import Tensor, nn
from .nn_modules import TransformerBlock, Embedding, RMSNorm, Linear, RotaryPositionalEmbedding
from typing import Self
import einx


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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        d_k = d_model // num_heads

        rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_k,
            max_seq_len=context_length,
            device=device
        )

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.transformer_blocks = nn.Sequential(
            *(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    d_ff=d_ff,
                    rope=rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            )
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)


    def print_total_memory_usage(self, b: int = 1) -> None:
        print("P = v d_m + n_l (2d_m + h d_m (2d_k + d_v) + h d_m d_v + 3 d_ff d_m) + d_m + v d_m")
        print("A = n_l (4 b s d_m + 2 b s h (d_k + d_v) + 2 b h s^2 + 4 b s d_ff) + b s d_m + 2 b s v")
        print("Assuming d_k = d_v = d_m / h AND d_ff = (8/3) d_m, then:")
        print("\tP= (2v + 2n_l + 1) d_m + 12n_l d_m^2")
        print("\tA= b (2n_l h s^2 + ((56/3) n_l d_m + d_m + 2v) s)\n\t\t(dominant term switches from (56/3) n_l d_m b s to 2n_l h b s^2) when s ~ 9.3d_k")
        print("Weights: P, Gradients: P, AdamW state: 2P, Activations: A")

        v = self.vocab_size
        s = self.context_length
        dm = self.d_model
        nl = self.num_layers
        h = self.num_heads
        P = (2 * v + 2 * nl + 1) * dm + 12 * nl * (dm ** 2)
        A = int(b * (2 * nl * h * (s ** 2) + ((56/3) * nl * dm + dm + 2 * v) * s))
        print(f"Memory usage for our model (assuming d_ff == (8/3) d_m and a batch size of {b}):\n\t- P = {P:,} ({P*4/(1024 ** 3):,.3f}GiB)\n\t- A = {A:,} ({A*4/(1024 ** 3):,.3f}GiB)")
        total = 4 * P + A
        print(f"Total: (4P + A) = {total:,} floats ({total * 4 / (1024 ** 3):,.3f}GiB)")
        print()
        slope = A * 4 / (1024 ** 3)
        intercept = P * 16 / (1024 ** 3)
        print(f"The formula mapping batch_size -> memory for this model is\n\t mem = {slope:,.3f} * batch_size + {intercept:,.3f}")
        print(f"Given 80 GiB of memory, one could train with a batch size of {int((80 - intercept) / slope)}")

    def flops(self, seq: int) -> int:
        print("Assuming d_k = d_v = d_m / h and d_ff = (8/3) d_m:\n\tn_l (24s d_m^2 + 4s^2 d_m) + 2s v d_m FLOPS")
        out = sum(map(lambda tfb: tfb.flops(seq),self.transformer_blocks))
        out += 2 * seq * self.vocab_size * self.d_model  # Final linear layer
        return out

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
        weights: dict[str, Tensor],
    ) -> Self:
        tlm = cls(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

        for i in range(num_layers):
            bw: dict[str, Tensor] = {
                name: weights[f"layers.{i}.{name}"]
                for name in [
                    "attn.q_proj.weight",
                    "attn.k_proj.weight",
                    "attn.v_proj.weight",
                    "attn.output_proj.weight",
                    "ln1.weight",
                    "ln2.weight",
                    "ffn.w1.weight",
                    "ffn.w2.weight",
                    "ffn.w3.weight",
                ]
            }
            tlm.transformer_blocks[i] = TransformerBlock.from_weights(
                d_model, num_heads, context_length, d_ff, rope_theta, bw
            )

        Wemb = weights["token_embeddings.weight"]
        Wln_final = weights["ln_final.weight"]
        Wlm_head = weights["lm_head.weight"]

        tlm.embedding.load_state_dict({"W": Wemb})
        tlm.ln_final.load_state_dict({"gain": Wln_final})
        tlm.lm_head.load_state_dict({"W": Wlm_head})

        return tlm

    def forward(self, indices: Tensor, return_logits: bool = True) -> Tensor:
        x = self.embedding.forward(indices)

        x = self.transformer_blocks.forward(x)

        y = self.lm_head.forward(self.ln_final.forward(x))

        if return_logits:
            return y

        return einx.softmax("... seq [vocab]",y)

if __name__ == "__main__":
    tlm = TransformerLM(
        50_257,
        1_024,
        1_600,
        48,
        25,
        4_288,
        10_000
    )
    print()
    print()
    tlm.print_total_memory_usage()
    print()
    print()

    num_params = sum(p.numel() for p in tlm.parameters())
    print(f"GPT-2 XL-sized model has {num_params:,} parameters, which take a total of {num_params * 4 / (1024 ** 3):.3f}GiB to store")
    print()
    fp_TFLOP = tlm.flops(tlm.context_length) / 1e12
    print(f"Uses {fp_TFLOP:,.3f} TFLOP per forward pass")

    TFLOPS = 495
    MFU = 0.5
    actual_TFLOPS = TFLOPS * MFU
    batch_size = 1_024
    steps = 400_000

    total_training_flops = steps * batch_size * fp_TFLOP * 3

    total_s = total_training_flops / actual_TFLOPS
    d, rem = divmod(total_s, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)

    print(f"Given {TFLOPS} TFLOPS and MFU={MFU}, would take {int(d)} days, {int(h)}h {int(m)}m {s:.1f}s to train {steps:,} steps with batch size {batch_size:,}")

    # from src.assignment1.bpe.tokenizer import Tokenizer
    # import pickle

    # with open("results/bpe_train_tiny_stories/special_tokens.pkl","rb") as f:
    #     special_tokens = pickle.load(f)

    # tokenizer = Tokenizer.from_files(
    #     "results/bpe_train_tiny_stories/vocab.pkl",
    #     "results/bpe_train_tiny_stories/merges.pkl",
    #     special_tokens,
    # )

    # tlm = TransformerLM(
    #     vocab_size=len(tokenizer.vocab),
    #     context_length=512,
    #     d_model=256,
    #     num_layers=16,
    #     num_heads=8,
    #     d_ff=2_48,
    #     rope_theta=10_000
    # )

    # print(f"num flops: {tlm.flops(1_024):,}")

    # prefix = input("Text to complete: ")

    # while True:
    #     encoding = torch.tensor(tokenizer.encode(prefix),dtype=torch.long)
    #     dists = tlm.forward(encoding,return_logits=False)
    #     next_token_id = dists[-1].argmax().item()
    #     next_token = tokenizer.vocab[next_token_id].decode('utf-8',errors='ignore')
    #     if next_token in special_tokens:
    #         print()
    #         print("finished!")
    #         break
    #     else:
    #         print(next_token,end="",flush=True)
