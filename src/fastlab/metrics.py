"""Reconstruction-quality and compression metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReconstructionError:
    mse: float
    mae: float
    max_abs: float
    # Error relative to each dimension's own spread, averaged over dimensions.
    # Raw MSE is dominated by whichever joint happens to have the largest units;
    # this keeps a gripper and a base rotation comparable.
    normalized_rmse: float


def reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> ReconstructionError:
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(reconstructed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    err = a - b
    flat = a.reshape(-1, a.shape[-1])
    spread = flat.std(axis=0)
    spread = np.where(spread < 1e-8, 1.0, spread)
    per_dim_rmse = np.sqrt((err.reshape(-1, err.shape[-1]) ** 2).mean(axis=0))

    return ReconstructionError(
        mse=float(np.mean(err**2)),
        mae=float(np.mean(np.abs(err))),
        max_abs=float(np.max(np.abs(err))),
        normalized_rmse=float(np.mean(per_dim_rmse / spread)),
    )


@dataclass(frozen=True)
class CompressionStats:
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    naive_tokens: int
    mean_sparsity: float

    @property
    def compression_vs_naive(self) -> float:
        """How many times fewer tokens than per-timestep binning."""
        return self.naive_tokens / self.mean_tokens if self.mean_tokens else float("inf")


def compression_stats(
    token_lists, time_horizon: int, action_dim: int, sparsities=None
) -> CompressionStats:
    counts = np.asarray([len(t) for t in token_lists], dtype=np.float64)
    return CompressionStats(
        mean_tokens=float(counts.mean()),
        median_tokens=float(np.median(counts)),
        p95_tokens=float(np.percentile(counts, 95)),
        naive_tokens=time_horizon * action_dim,
        mean_sparsity=float(np.mean(sparsities)) if sparsities is not None else float("nan"),
    )
