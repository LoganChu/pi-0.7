"""The frequency-space half of FAST: DCT -> quantize -> flatten.

This is deliberately written as free functions on plain arrays so that every
intermediate (the raw coefficient matrix, the sparse integer matrix, the
flattened stream) can be pulled out and plotted. `fastlab.tokenizer` stacks
these together with BPE into the full pipeline.

Shapes throughout:
    chunk        (T, D)  -- T timesteps, D action dimensions
    coefficients (T, D)  -- row f is the f-th DCT frequency, per dimension
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.fft import dct as _dct
from scipy.fft import idct as _idct

FlattenOrder = Literal["freq_major", "dim_major"]

DEFAULT_SCALE = 10.0
DEFAULT_FLATTEN: FlattenOrder = "freq_major"


def forward_dct(chunk: np.ndarray) -> np.ndarray:
    """DCT-II along the time axis, applied independently per action dimension.

    Orthonormal so that the transform preserves energy: the coefficient
    magnitudes are directly comparable to the action magnitudes, which is what
    makes a single global `scale` work across dimensions.
    """
    arr = _as_chunk(chunk)
    return _dct(arr, type=2, axis=-2, norm="ortho")


def inverse_dct(coefficients: np.ndarray) -> np.ndarray:
    """Inverse of `forward_dct`."""
    arr = _as_chunk(coefficients)
    return _idct(arr, type=2, axis=-2, norm="ortho")


def quantize(coefficients: np.ndarray, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Round scaled coefficients to integers.

    This is the only lossy step in FAST. Coefficients whose magnitude falls
    below 1/(2*scale) round to exactly zero -- which is the entire point, since
    smooth robot trajectories put almost all of their energy in the first few
    frequencies and the rest of the matrix collapses to zeros that BPE can then
    merge into single tokens.

    Larger `scale` -> finer resolution -> lower reconstruction error, more
    non-zero coefficients, and more tokens per chunk.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return np.round(np.asarray(coefficients, dtype=np.float64) * scale).astype(np.int32)


def dequantize(quantized: np.ndarray, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Inverse of `quantize` (up to the rounding loss)."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return np.asarray(quantized, dtype=np.float64) / scale


def flatten(matrix: np.ndarray, order: FlattenOrder = DEFAULT_FLATTEN) -> np.ndarray:
    """Flatten a (T, D) coefficient matrix into a 1-D integer stream.

    The ordering matters a lot for how well BPE compresses, because BPE can only
    merge *adjacent* symbols:

    - "freq_major": [f0d0, f0d1, ..., f0dD, f1d0, ...] -- all dimensions at the
      lowest frequency first. Every high frequency is near-zero across *all*
      dimensions, so the zeros collect into one contiguous run at the tail.
    - "dim_major":  [d0f0, d0f1, ..., d0fT, d1f0, ...] -- one dimension's whole
      spectrum, then the next. Produces D shorter zero runs instead of one long
      one.

    The FAST paper describes low-frequency-first ordering, i.e. "freq_major".
    experiments/02 measures what the choice actually costs.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (T, D) matrix, got {arr.shape}")
    if order == "freq_major":
        return arr.reshape(-1)
    if order == "dim_major":
        return arr.T.reshape(-1)
    raise ValueError(f"unknown flatten order: {order!r}")


def unflatten(
    stream: np.ndarray,
    time_horizon: int,
    action_dim: int,
    order: FlattenOrder = DEFAULT_FLATTEN,
) -> np.ndarray:
    """Inverse of `flatten`.

    A decoded stream may be short (trailing zeros dropped) or long (a corrupt
    generation); it is zero-padded or truncated to fit exactly.
    """
    expected = time_horizon * action_dim
    arr = np.asarray(stream).reshape(-1)
    if arr.size < expected:
        arr = np.concatenate([arr, np.zeros(expected - arr.size, dtype=arr.dtype)])
    elif arr.size > expected:
        arr = arr[:expected]

    if order == "freq_major":
        return arr.reshape(time_horizon, action_dim)
    if order == "dim_major":
        return arr.reshape(action_dim, time_horizon).T
    raise ValueError(f"unknown flatten order: {order!r}")


def drop_trailing_zeros(stream: np.ndarray) -> np.ndarray:
    """Trim the tail of zeros from a flattened stream.

    FAST relies on this: the decoder knows T and D, so any coefficient not
    transmitted is implicitly zero. On smooth trajectories this alone removes
    most of the stream before BPE ever runs.
    """
    arr = np.asarray(stream).reshape(-1)
    nonzero = np.flatnonzero(arr)
    return arr[: nonzero[-1] + 1] if nonzero.size else arr[:0]


def sparsity(quantized: np.ndarray) -> float:
    """Fraction of quantized coefficients that are exactly zero."""
    arr = np.asarray(quantized)
    return float(np.mean(arr == 0)) if arr.size else 0.0


def _as_chunk(chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"expected at least (T, D), got {arr.shape}")
    return arr
