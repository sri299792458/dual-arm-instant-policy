from __future__ import annotations
import math
import torch
from torch import nn

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for a scalar diffusion time t in [0,1] or int steps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class PositionalEncoder(nn.Module):
    """Fourier features / NeRF positional encoding for 3D points or deltas."""
    def __init__(self, in_dim: int = 3, num_freqs: int = 10, log_space: bool = True,
                 add_original_x: bool = True, scale: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.log_space = log_space
        self.add_original_x = add_original_x
        self.scale = scale

        if log_space:
            self.freq_bands = 2.0 ** torch.arange(num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)

        out = 0
        if add_original_x:
            out += in_dim
        out += 2 * in_dim * num_freqs
        self.d_output = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        x = x * self.scale
        fb = self.freq_bands.to(device=x.device, dtype=x.dtype)
        # (..., 1, in_dim) * (num_freqs,) -> (..., num_freqs, in_dim)
        xb = x.unsqueeze(-2) * fb.view(*([1] * (x.dim() - 1)), -1, 1)
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        pe = torch.cat([sin, cos], dim=-1).reshape(*x.shape[:-1], -1)
        if self.add_original_x:
            pe = torch.cat([x, pe], dim=-1)
        return pe
