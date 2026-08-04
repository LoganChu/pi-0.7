"""The full FAST pipeline, assembled from `normalize`, `dct` and `bpe`.

    raw actions -> quantile norm -> DCT -> quantize -> flatten
                -> drop trailing zeros -> BPE -> tokens

`encode_trace` returns every intermediate from a single chunk, which is what the
experiment scripts plot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import dct as dct_ops
from . import normalize as norm_ops
from .bpe import BPECodec
from .dct import DEFAULT_FLATTEN, DEFAULT_SCALE, FlattenOrder


@dataclass(frozen=True)
class FastConfig:
    scale: float = DEFAULT_SCALE
    flatten_order: FlattenOrder = DEFAULT_FLATTEN
    vocab_size: int = 1024
    min_frequency: int = 2
    normalize: bool = True
    low_q: float = norm_ops.DEFAULT_LOW_Q
    high_q: float = norm_ops.DEFAULT_HIGH_Q


@dataclass
class EncodeTrace:
    """Every intermediate of a single chunk's trip through the pipeline."""

    raw: np.ndarray  # (T, D) input actions
    normalized: np.ndarray  # (T, D) in [-1, 1]
    coefficients: np.ndarray  # (T, D) float DCT-II coefficients
    quantized: np.ndarray  # (T, D) int32
    stream: np.ndarray  # (T*D,) flattened ints
    trimmed: np.ndarray  # stream with trailing zeros dropped
    tokens: list[int]  # final BPE token ids

    @property
    def sparsity(self) -> float:
        return dct_ops.sparsity(self.quantized)

    @property
    def compression_ratio(self) -> float:
        """Raw coefficients per emitted token."""
        return self.stream.size / len(self.tokens) if self.tokens else float("inf")


class FastTokenizer:
    """A FAST tokenizer fitted to one action distribution.

    FAST is dataset-specific by design: the quantile bounds and the BPE merges
    both encode what the action distribution actually looks like. The published
    `physical-intelligence/fast` checkpoint is the universal variant, fitted
    across many robots; `fit` here reproduces the per-dataset training path.
    """

    def __init__(
        self,
        config: FastConfig,
        bpe: BPECodec,
        stats: norm_ops.QuantileStats | None,
        time_horizon: int,
        action_dim: int,
    ):
        self.config = config
        self._bpe = bpe
        self._stats = stats
        self.time_horizon = time_horizon
        self.action_dim = action_dim

    # -- construction ----------------------------------------------------

    @classmethod
    def fit(cls, chunks: np.ndarray, config: FastConfig | None = None) -> "FastTokenizer":
        """Fit normalization bounds and the BPE vocabulary on a chunk dataset.

        Args:
            chunks: (N, T, D) raw action chunks.
            config: pipeline hyperparameters.
        """
        cfg = config or FastConfig()
        arr = _as_batch(chunks)
        _, time_horizon, action_dim = arr.shape

        stats = (
            norm_ops.fit(arr, low_q=cfg.low_q, high_q=cfg.high_q)
            if cfg.normalize
            else None
        )
        normalized = norm_ops.apply(arr, stats) if stats is not None else arr

        streams = [
            dct_ops.drop_trailing_zeros(
                dct_ops.flatten(
                    dct_ops.quantize(dct_ops.forward_dct(chunk), cfg.scale),
                    cfg.flatten_order,
                )
            )
            for chunk in normalized
        ]
        bpe = BPECodec.train(
            streams, vocab_size=cfg.vocab_size, min_frequency=cfg.min_frequency
        )
        return cls(cfg, bpe, stats, time_horizon, action_dim)

    # -- encode / decode -------------------------------------------------

    def encode(self, chunk: np.ndarray) -> list[int]:
        return self.encode_trace(chunk).tokens

    def encode_trace(self, chunk: np.ndarray) -> EncodeTrace:
        """Encode one chunk, keeping every intermediate for inspection."""
        raw = np.asarray(chunk, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError(f"expected a single (T, D) chunk, got {raw.shape}")

        normalized = (
            norm_ops.apply(raw, self._stats) if self._stats is not None else raw
        )
        coefficients = dct_ops.forward_dct(normalized)
        quantized = dct_ops.quantize(coefficients, self.config.scale)
        stream = dct_ops.flatten(quantized, self.config.flatten_order)
        trimmed = dct_ops.drop_trailing_zeros(stream)
        tokens = self._bpe.encode(trimmed)
        return EncodeTrace(
            raw=raw,
            normalized=normalized,
            coefficients=coefficients,
            quantized=quantized,
            stream=stream,
            trimmed=trimmed,
            tokens=tokens,
        )

    def decode(
        self,
        tokens: Sequence[int],
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        """Reconstruct a (T, D) action chunk from token ids."""
        t = time_horizon or self.time_horizon
        d = action_dim or self.action_dim
        stream = self._bpe.decode(tokens)
        quantized = dct_ops.unflatten(stream, t, d, self.config.flatten_order)
        coefficients = dct_ops.dequantize(quantized, self.config.scale)
        normalized = dct_ops.inverse_dct(coefficients)
        if self._stats is None:
            return normalized
        return norm_ops.invert(normalized, self._stats)

    def encode_batch(self, chunks: np.ndarray) -> list[list[int]]:
        return [self.encode(c) for c in _as_batch(chunks)]

    def decode_batch(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        return np.stack([self.decode(t) for t in batch])

    # -- introspection ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._bpe.vocab_size

    @property
    def bpe(self) -> BPECodec:
        return self._bpe

    @property
    def stats(self) -> norm_ops.QuantileStats | None:
        return self._stats

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._bpe.save(path)
        meta = {
            "config": asdict(self.config),
            "time_horizon": self.time_horizon,
            "action_dim": self.action_dim,
            "stats": (
                None
                if self._stats is None
                else {"low": self._stats.low.tolist(), "high": self._stats.high.tolist()}
            ),
        }
        (path / "fast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FastTokenizer":
        path = Path(path)
        meta = json.loads((path / "fast_meta.json").read_text(encoding="utf-8"))
        stats = (
            None
            if meta["stats"] is None
            else norm_ops.QuantileStats(
                low=np.asarray(meta["stats"]["low"]),
                high=np.asarray(meta["stats"]["high"]),
            )
        )
        return cls(
            config=FastConfig(**meta["config"]),
            bpe=BPECodec.load(path),
            stats=stats,
            time_horizon=meta["time_horizon"],
            action_dim=meta["action_dim"],
        )


def _as_batch(chunks: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunks, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T, D) action chunks, got {arr.shape}")
    return arr
