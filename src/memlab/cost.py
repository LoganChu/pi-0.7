"""What the memory encoder costs, counted rather than timed.

Wall-clock on a laptop CPU is dominated by kernel launch overhead and thread
scheduling, which tells you about PyTorch rather than about the architecture. An
analytic FLOP count does not move when the machine is busy, so the shape of the
curve in `experiments/07` is a property of MEM's design.

Everything is counted as FLOPs = 2 x MACs, the usual convention. Layer norms,
GELU and softmax are excluded: they are O(tokens x dim) against the O(tokens x
dim^2) terms that follow, and including them would not shift any curve here.

Three cost regimes are modelled, and the gap between them is the whole point:

    joint      one attention over all T x N tokens          quadratic in history
    factorized space and time attended separately           linear in history
    MEM        factorized, and history dropped up top       linear, but only the
                                                            layers below the cut

MEM's saving is therefore a *slope*, not a ceiling: with half the depth above
the cut, history costs half as much per frame. Note also that the quadratic
penalty on joint attention is milder at this scale than at MEM's -- 64 patches
per frame rather than several hundred -- so panel (a) of experiment 07
understates how bad the joint baseline gets on real images.
"""

from __future__ import annotations

from dataclasses import dataclass

from .encoder import EncoderConfig


@dataclass(frozen=True)
class Cost:
    """FLOPs for one forward pass over one sample, split by where they go."""

    patch_embed: float
    spatial_attn: float
    temporal_attn: float
    mlp: float

    @property
    def total(self) -> float:
        return self.patch_embed + self.spatial_attn + self.temporal_attn + self.mlp


def tokens_per_layer(cfg: EncoderConfig, history: int, drop: bool = True) -> list[int]:
    """Timesteps entering each layer. The step down is the dropping."""
    cut = cfg.drop_history_after if drop else cfg.depth
    return [history if i < cut else 1 for i in range(cfg.depth)]


def flops(cfg: EncoderConfig, history: int, drop: bool = True) -> Cost:
    """Cost of MEM's factorized encoder over `history` frames."""
    n, d = cfg.n_patches, cfg.dim
    temporal = set(cfg.temporal_layers)

    patch = 2.0 * history * n * (cfg.patch_px**2 * 3 * d)
    spatial = temporal_cost = mlp = 0.0

    for i, t in enumerate(tokens_per_layer(cfg, history, drop)):
        spatial += 2.0 * t * (4 * n * d * d + 2 * n * n * d)
        mlp += 2.0 * (2.0 * cfg.mlp_ratio * t * n * d * d)
        if i in temporal and t > 1:
            # Per spatial position, attention runs over the time axis.
            temporal_cost += 2.0 * n * (4 * t * d * d + 2 * t * t * d)

    return Cost(patch_embed=patch, spatial_attn=spatial, temporal_attn=temporal_cost, mlp=mlp)


def joint_flops(cfg: EncoderConfig, history: int) -> Cost:
    """Cost if you skipped factorizing and attended over every token at once.

    This is the baseline MEM's design is avoiding: the quadratic term is
    (T x N)^2 rather than N^2 per frame, so it blows up in history length.
    """
    n, d = cfg.n_patches, cfg.dim
    length = history * n

    patch = 2.0 * history * n * (cfg.patch_px**2 * 3 * d)
    attn = cfg.depth * 2.0 * (4 * length * d * d + 2 * length * length * d)
    mlp = cfg.depth * 2.0 * (2.0 * cfg.mlp_ratio * length * d * d)
    return Cost(patch_embed=patch, spatial_attn=attn, temporal_attn=0.0, mlp=mlp)


def measure_grid(
    conditions: list[tuple[str, EncoderConfig, int]],
    batch: int = 4,
    reps: int = 5,
    threads: int = 4,
) -> dict[str, float]:
    """Time several encoder configurations, returning seconds per sample.

    Four details keep this from measuring PyTorch instead of the architecture,
    each one added after the previous version reported something impossible:

    - **Batch 4, not 1.** At batch 1 these tensors are small enough that kernel
      launch and thread hand-off dominate and the FLOP term vanishes into noise.
      Batch 1 produced a *falling* curve in history length.
    - **Minimum, not mean.** Every scheduler interruption adds time and none
      subtracts it, so the fastest run is the least contaminated estimate.
    - **Interleaved, not blocked.** All conditions are timed once per round,
      round after round. Timing one condition to completion before starting the
      next lets a background spike land entirely on one of them -- which is how
      T=12 came out slower than T=18.
    - **Fixed thread count**, so the pool does not resize between conditions.

    Even so, this is the number that drifts with whatever else the laptop is
    doing. `flops` is the claim; this is only the sanity check on it.
    """
    import time

    import torch

    from .encoder import MemEncoder

    torch.set_num_threads(threads)
    built = []
    for key, cfg, history in conditions:
        model = MemEncoder(cfg).eval()
        frames = torch.randint(
            0, 255, (batch, history, cfg.frame_px, cfg.frame_px, 3), dtype=torch.uint8
        )
        built.append((key, model, frames))

    best = {key: float("inf") for key, _, _ in conditions}
    with torch.no_grad():
        for _, model, frames in built:
            model(frames)  # warm the allocator and kernel selection
        for _ in range(reps):
            for key, model, frames in built:
                start = time.perf_counter()
                model(frames)
                best[key] = min(best[key], (time.perf_counter() - start) / batch)
    return best


def measure_seconds(cfg: EncoderConfig, history: int, **kwargs) -> float:
    """Convenience wrapper around `measure_grid` for a single configuration."""
    return measure_grid([("only", cfg, history)], **kwargs)["only"]
