"""Tokenization baselines that FAST is measured against.

The comparison that matters is against naive per-timestep binning, which is what
most pre-FAST autoregressive VLAs used (RT-1, OpenVLA). Its token count is fixed
at T*D regardless of how smooth the trajectory is -- it cannot exploit temporal
redundancy at all, which is exactly why it degrades on high-frequency dexterous
data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import normalize as norm_ops


@dataclass
class NaiveBinningTokenizer:
    """One token per (timestep, action dimension), from uniform bins in [-1, 1].

    This is the RT-1/OpenVLA-style scheme: discretize each scalar independently.
    """

    n_bins: int
    stats: norm_ops.QuantileStats | None

    @classmethod
    def fit(
        cls, chunks: np.ndarray, n_bins: int = 256, normalize: bool = True
    ) -> "NaiveBinningTokenizer":
        stats = norm_ops.fit(chunks) if normalize else None
        return cls(n_bins=n_bins, stats=stats)

    def encode(self, chunk: np.ndarray) -> list[int]:
        arr = np.asarray(chunk, dtype=np.float64)
        if self.stats is not None:
            arr = norm_ops.apply(arr, self.stats)
        idx = np.floor((arr + 1.0) / 2.0 * self.n_bins).astype(np.int64)
        idx = np.clip(idx, 0, self.n_bins - 1)
        return idx.reshape(-1).tolist()

    def decode(self, tokens, time_horizon: int, action_dim: int) -> np.ndarray:
        idx = np.asarray(tokens, dtype=np.int64).reshape(time_horizon, action_dim)
        # Bin centers, not edges: halves the worst-case quantization error.
        arr = (idx + 0.5) / self.n_bins * 2.0 - 1.0
        if self.stats is None:
            return arr
        return norm_ops.invert(arr, self.stats)

    @property
    def vocab_size(self) -> int:
        return self.n_bins


def naive_token_count(time_horizon: int, action_dim: int) -> int:
    """Naive binning always emits exactly this many tokens."""
    return time_horizon * action_dim
