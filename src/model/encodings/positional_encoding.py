import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor


class PositionalEncoding(nn.Module):
    """For the sake of simplicity, this encodes values in the range [0, 1]."""

    frequencies: Float[Tensor, "frequency phase"]
    phases: Float[Tensor, "frequency phase"]

    def __init__(self, num_octaves: int):
        super().__init__()
        octaves = torch.arange(num_octaves).float()

        # The lowest frequency has a period of 1.
        frequencies = 2 * torch.pi * 2**octaves
        frequencies = repeat(frequencies, "f -> f p", p=2)
        self.register_buffer("frequencies", frequencies, persistent=False)

        # Choose the phases to match sine and cosine.
        phases = torch.tensor([0, 0.5 * torch.pi], dtype=torch.float32)
        phases = repeat(phases, "p -> f p", f=num_octaves)
        self.register_buffer("phases", phases, persistent=False)

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        samples = einsum(samples, self.frequencies, "... d, f p -> ... d f p")
        return rearrange(torch.sin(samples + self.phases), "... d f p -> ... (d f p)")

    def d_out(self, dimensionality: int):
        return self.frequencies.numel() * dimensionality
