"""MEM's short-horizon memory encoder, at 2-D scale.

This is the piece of MEM that is purely architectural, so it shrinks without
losing anything. Three claims from the paper are reproduced here exactly, and
each one is measurable in `experiments/07` and `experiments/08`:

1. Attention is factorized. Spatial attention runs every layer, as in any ViT.
   Temporal attention runs every `temporal_every`-th layer and connects the same
   spatial position across timesteps.
2. History tokens are dropped in the upper layers -- past this depth, only the
   current timestep's patches propagate. Cost stops scaling with history length
   at the point where the history has already been read.
3. It adds no learnable parameters over a single-image ViT.

Claim 3 is what pins down the design, and the paper does not spell out how it is
achieved. Two choices here follow from it, and both are flagged as *readings*
rather than transcriptions:

- Temporal attention **reuses the spatial attention's weights** (and its norm),
  applied over the time axis instead of the space axis. A separately
  parameterized temporal block would add parameters, which claim 3 forbids.
  Set `share_time_space_weights=False` to measure what that would have cost.
- Time is encoded with a **fixed sinusoidal** embedding, since a learned one
  would also be new parameters. It is indexed by *lag* rather than absolute
  step, so the current frame is always lag 0 whatever the history length.

Reference: Torne et al., "MEM: Multi-Scale Embodied Memory for Vision Language
Action Models" (2026), sec. III-C. https://arxiv.org/abs/2603.03596
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    frame_px: int = 64
    patch_px: int = 8
    dim: int = 192
    depth: int = 12
    heads: int = 3
    mlp_ratio: float = 4.0
    temporal_every: int = 4          # 0 disables temporal attention entirely
    drop_history_after: int = 6      # layers beyond this see the current frame only
    share_time_space_weights: bool = True

    @property
    def n_patches(self) -> int:
        side = self.frame_px // self.patch_px
        return side * side

    @property
    def temporal_layers(self) -> tuple[int, ...]:
        """Layer indices that carry temporal attention."""
        if self.temporal_every <= 0:
            return ()
        return tuple(i for i in range(self.depth) if (i + 1) % self.temporal_every == 0)

    @property
    def effective_temporal_layers(self) -> tuple[int, ...]:
        """Temporal layers that still see a history to mix.

        A temporal layer sitting above `drop_history_after` has a single
        timestep to attend over, so it is an identity. Worth naming, because the
        two hyperparameters interact in a way that is easy to miss: with the
        defaults, only one of the three temporal layers does anything.
        """
        return tuple(i for i in self.temporal_layers if i < self.drop_history_after)


class Attention(nn.Module):
    """Plain multi-head self-attention over whatever axis it is handed."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, length, dim))


class Block(nn.Module):
    """One ViT layer, optionally also mixing along time before mixing space."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden), nn.GELU(), nn.Linear(hidden, cfg.dim)
        )
        if cfg.share_time_space_weights:
            # The whole of claim 3: temporal mixing is the same operator,
            # pointed at the other axis. Assigning the modules rather than
            # copying them means they are one set of parameters, not two.
            self.norm_t, self.attn_t = self.norm1, self.attn
        else:
            self.norm_t = nn.LayerNorm(cfg.dim)
            self.attn_t = Attention(cfg.dim, cfg.heads)

    def forward(self, x: torch.Tensor, temporal: bool) -> torch.Tensor:
        b, t, n, d = x.shape
        if temporal and t > 1:
            h = self.norm_t(x).permute(0, 2, 1, 3).reshape(b * n, t, d)
            h = self.attn_t(h).reshape(b, n, t, d).permute(0, 2, 1, 3)
            x = x + h
        h = self.norm1(x).reshape(b * t, n, d)
        x = x + self.attn(h).reshape(b, t, n, d)
        return x + self.mlp(self.norm2(x))


class MemEncoder(nn.Module):
    """Encode a short frame history into tokens for the current timestep.

    Input is `(B, T, H, W, 3)` uint8 or float frames, oldest first, so frame
    `-1` is the present. Output is `(B, n_patches, dim)` -- the current frame's
    patches, having read the history.
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.patch = nn.Conv2d(
            3, self.cfg.dim, kernel_size=self.cfg.patch_px, stride=self.cfg.patch_px
        )
        self.pos = nn.Parameter(torch.zeros(1, 1, self.cfg.n_patches, self.cfg.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.depth))
        self.norm = nn.LayerNorm(self.cfg.dim)
        self._temporal = set(self.cfg.temporal_layers)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, t = frames.shape[:2]
        x = _to_float(frames).reshape(b * t, *frames.shape[2:]).permute(0, 3, 1, 2)
        x = self.patch(x).flatten(2).transpose(1, 2).reshape(b, t, cfg.n_patches, cfg.dim)
        x = x + self.pos + _lag_embedding(t, cfg.dim, x.device, x.dtype)

        for i, block in enumerate(self.blocks):
            if i == cfg.drop_history_after:
                x = x[:, -1:]  # history has been read; carry the present only
            x = block(x, temporal=i in self._temporal)
        return self.norm(x[:, -1])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _to_float(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        return frames.float().div_(127.5).sub_(1.0)
    return frames


def _lag_embedding(t: int, dim: int, device, dtype) -> torch.Tensor:
    """Fixed sinusoidal embedding indexed by lag, so the present is always 0.

    Parameter-free by necessity (see the module docstring), and lag-relative so
    that changing the history length does not relabel the current frame.
    """
    lag = torch.arange(t - 1, -1, -1, device=device, dtype=torch.float32)
    freq = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angle = lag[:, None] * freq[None, :]
    emb = torch.zeros(t, dim, device=device, dtype=torch.float32)
    emb[:, 0::2], emb[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return emb.to(dtype)[None, :, None, :]
