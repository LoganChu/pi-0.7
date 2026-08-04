"""Per-dimension quantile normalization of action chunks.

FAST expects action chunks pre-normalized to roughly [-1, 1]. Quantile
normalization (rather than min/max) is recommended because robot action
distributions have heavy tails: a handful of outlier timesteps would otherwise
compress the entire useful range into a sliver near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LOW_Q = 1.0
DEFAULT_HIGH_Q = 99.0
_EPS = 1e-8


@dataclass(frozen=True)
class QuantileStats:
    """Per-dimension normalization bounds fit on a dataset of action chunks."""

    low: np.ndarray  # shape (D,)
    high: np.ndarray  # shape (D,)

    @property
    def action_dim(self) -> int:
        return int(self.low.shape[0])


def fit(
    chunks: np.ndarray,
    low_q: float = DEFAULT_LOW_Q,
    high_q: float = DEFAULT_HIGH_Q,
) -> QuantileStats:
    """Fit per-dimension quantiles over a dataset of action chunks.

    Args:
        chunks: shape (N, T, D) or (T, D) of raw actions.
        low_q: lower percentile, in [0, 100].
        high_q: upper percentile, in [0, 100].

    Returns:
        QuantileStats with per-dimension bounds.
    """
    arr = _as_batch(chunks)
    if not 0.0 <= low_q < high_q <= 100.0:
        raise ValueError(f"require 0 <= low_q < high_q <= 100, got {low_q}, {high_q}")

    flat = arr.reshape(-1, arr.shape[-1])
    low = np.percentile(flat, low_q, axis=0)
    high = np.percentile(flat, high_q, axis=0)

    # A constant dimension (e.g. an unused joint) has low == high; widen it so
    # normalization maps it to 0 instead of dividing by zero.
    degenerate = (high - low) < _EPS
    low = np.where(degenerate, low - 0.5, low)
    high = np.where(degenerate, high + 0.5, high)
    return QuantileStats(low=low.astype(np.float64), high=high.astype(np.float64))


def apply(chunks: np.ndarray, stats: QuantileStats, clip: bool = True) -> np.ndarray:
    """Map raw actions into [-1, 1] using fitted quantile bounds."""
    scale = stats.high - stats.low
    out = 2.0 * (chunks - stats.low) / scale - 1.0
    return np.clip(out, -1.0, 1.0) if clip else out


def invert(chunks_norm: np.ndarray, stats: QuantileStats) -> np.ndarray:
    """Map normalized actions in [-1, 1] back to raw action units."""
    scale = stats.high - stats.low
    return (chunks_norm + 1.0) / 2.0 * scale + stats.low


def _as_batch(chunks: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunks, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T, D) or (T, D) action chunks, got {arr.shape}")
    return arr
