"""fastlab -- a transparent reimplementation of FAST action tokenization.

FAST (Frequency-space Action Sequence Tokenization) is the action tokenizer
behind pi-0-FAST and the autoregressive half of the later pi-0.x models. The
pipeline is:

    quantile-normalize -> DCT-II over time -> quantize -> flatten -> BPE

Every stage is exposed separately here so it can be plotted and ablated, rather
than hidden behind a single encode() call.

Reference: Pertsch et al., "FAST: Efficient Action Tokenization for
Vision-Language-Action Models" (2025). https://www.pi.website/research/fast
"""

from .baselines import NaiveBinningTokenizer, naive_token_count
from .metrics import (
    CompressionStats,
    ReconstructionError,
    compression_stats,
    reconstruction_error,
)
from .tokenizer import EncodeTrace, FastConfig, FastTokenizer

__all__ = [
    "FastTokenizer",
    "FastConfig",
    "EncodeTrace",
    "NaiveBinningTokenizer",
    "naive_token_count",
    "reconstruction_error",
    "ReconstructionError",
    "compression_stats",
    "CompressionStats",
]

__version__ = "0.1.0"
