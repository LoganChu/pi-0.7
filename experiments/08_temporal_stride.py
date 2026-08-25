"""How often does temporal attention actually need to run?

MEM puts temporal attention on every 4th layer. That number is stated, not
derived, so it is worth asking what it buys and what it costs -- and whether 4
interacts with the *other* history hyperparameter in a way the figure makes
obvious but prose does not.

It does. A temporal layer above the drop point has one timestep to attend over,
so it is an identity. `temporal_every` and `drop_history_after` therefore cannot
be chosen independently: at the paper's stride of 4 with a cut at layer 6, two
of the three temporal layers are dead weight, and at a stride of 12 the encoder
has no working temporal path at all despite nominally having one.

WHAT PANEL (b) IS AND IS NOT
----------------------------
The weights here are random. Perturbing a past frame and measuring how far the
current representation moves therefore measures the *architecture's* capacity to
carry information across time -- whether a path exists at all, and how strong it
is at initialization. It says nothing about whether a trained model would choose
to use that path. Reading it as "the model relies on lag 4" would be wrong; read
it as "lag 4 can reach the output, lag 6 structurally cannot."

The control makes this concrete: with temporal attention off, past-frame
sensitivity is exactly zero, because no other component in the encoder mixes
across time.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from fastlab.viz import (
    CATEGORICAL,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL_CMAP,
    header,
    save,
    use_style,
)
from memlab.cost import flops
from memlab.encoder import EncoderConfig, MemEncoder
from memlab.oracle import collect

HISTORY = 8
STRIDES = [1, 2, 4, 6, 12]
SEEDS = (0, 1)
N_WINDOWS = 6


def episode_windows(history: int, n: int, seed: int = 3) -> torch.Tensor:
    """`n` windows of `history` consecutive frames from a real demonstration.

    Real frames rather than noise: the perturbation should be a plausible visual
    change -- the agent somewhere else in the kitchen -- not an off-distribution
    input the patch embedding has never seen.
    """
    episode = collect("find_object", 1, seed=seed)[0]
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(episode) - history, size=n, replace=False)
    return torch.from_numpy(
        np.stack([episode.frames[s : s + history] for s in starts])
    )


def lag_sensitivity(cfg: EncoderConfig, frames: torch.Tensor) -> np.ndarray:
    """Relative change in the output when the frame at each lag is swapped out.

    Lag 0 is the current frame, lag k is k steps back. The replacement comes
    from another window at the same lag, so the swapped-in frame is as
    in-distribution as the one it replaces.
    """
    b, t = frames.shape[:2]
    other = torch.roll(torch.arange(b), 1)
    out = np.zeros(t)
    for seed in SEEDS:
        torch.manual_seed(seed)
        encoder = MemEncoder(cfg).eval()
        with torch.no_grad():
            base = encoder(frames)
            denom = float(base.norm())
            for lag in range(t):
                perturbed = frames.clone()
                perturbed[:, t - 1 - lag] = frames[other, t - 1 - lag]
                out[lag] += float((encoder(perturbed) - base).norm()) / denom
    return out / len(SEEDS)


def main() -> None:
    use_style()
    frames = episode_windows(HISTORY, N_WINDOWS)
    configs = {s: EncoderConfig(temporal_every=s) for s in STRIDES}
    configs[0] = EncoderConfig(temporal_every=0)

    sensitivity = {}
    for stride in [*STRIDES, 0]:
        cfg = configs[stride]
        sensitivity[stride] = lag_sensitivity(cfg, frames)
        print(f"  every {stride or '-':>2}  temporal layers {cfg.temporal_layers} "
              f"effective {cfg.effective_temporal_layers}  "
              f"past-lag sensitivity {sensitivity[stride][1:].sum():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) what temporal attention costs -------------------------------------
    ax = axes[0]
    totals = [flops(configs[s], HISTORY).total / 1e9 for s in STRIDES]
    shares = [
        100.0 * flops(configs[s], HISTORY).temporal_attn / flops(configs[s], HISTORY).total
        for s in STRIDES
    ]
    ax.bar([str(s) for s in STRIDES], shares, color=CATEGORICAL[0], width=0.6)
    ax.set_xlabel("temporal attention every N-th layer")
    ax.set_ylabel("share of forward-pass FLOPs (%)")
    ax.set_title("a · Temporal attention is nearly free")
    for i, (share, total) in enumerate(zip(shares, totals)):
        ax.annotate(f"{share:.1f}%\n{total:.1f} GF", xy=(i, share), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_SECONDARY)
    ax.set_ylim(0, max(shares) * 1.45)
    ax.annotate(
        "4 and 6 cost the same:\none working layer each",
        xy=(0.52, 0.42), xycoords="axes fraction", fontsize=8.5, color=INK_MUTED,
    )

    # (b) how far back information can structurally reach --------------------
    #
    # Direct labels rather than a legend: six series over a narrow band of y
    # leaves a legend box nowhere to sit that is not on top of a curve.
    ax = axes[1]
    lags = np.arange(1, HISTORY)
    for slot, stride in enumerate([*STRIDES, 0]):
        curve = sensitivity[stride][1:]
        dead = not configs[stride].effective_temporal_layers
        ax.plot(lags, curve, color=CATEGORICAL[slot], marker="o", ms=5,
                ls="--" if stride == 12 else "-")
        if not dead:
            ax.annotate(f"every {stride}", xy=(lags[-1], curve[-1]), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=CATEGORICAL[slot])
    ax.annotate("every 12, and none", xy=(lags[-1], 0.0), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0.7, HISTORY + 1.4)
    ax.set_xticks(lags)
    ax.set_xlabel("lag (frames back from the present)")
    ax.set_ylabel("relative change in output")
    ax.set_title("b · Reach of a past frame, at init")
    ax.annotate(
        "at stride 12 the one temporal layer sits above the\n"
        "cut, so it is identical to having none at all",
        xy=(0.03, 0.10), xycoords="axes fraction", fontsize=8.5, color=INK_MUTED,
    )

    # (c) the interaction the paper leaves implicit ---------------------------
    ax = axes[2]
    strides = np.arange(1, 7)
    cuts = np.arange(0, 13)
    grid = np.array([
        [len(EncoderConfig(temporal_every=int(s), drop_history_after=int(c))
             .effective_temporal_layers) for c in cuts]
        for s in strides
    ])
    im = ax.imshow(grid, cmap=SEQUENTIAL_CMAP, origin="lower", aspect="auto",
                   extent=(-0.5, 12.5, 0.5, 6.5))
    default = EncoderConfig()
    ax.plot(default.drop_history_after, default.temporal_every, marker="s",
            ms=11, mfc="none", mec=CATEGORICAL[1], mew=2.0)
    ax.annotate("MEM default", xy=(default.drop_history_after, default.temporal_every),
                xytext=(8, 10), textcoords="offset points",
                fontsize=9, color=CATEGORICAL[1])
    ax.set_xticks(cuts[::2])
    ax.set_yticks(strides)
    ax.set_xlabel("drop_history_after (layer)")
    ax.set_ylabel("temporal attention every N-th layer")
    ax.set_title("c · Working temporal layers")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="layers that still see a history")

    header(
        fig,
        "Temporal attention is cheap; most of it is also inert",
        f"Only {len(default.effective_temporal_layers)} of "
        f"{len(default.temporal_layers)} temporal layers sit below the history cut, "
        f"for {shares[STRIDES.index(4)]:.1f}% of the forward pass. Panel (b) is measured "
        f"at random init — it shows which lags *can* reach the output, not which a "
        f"trained model uses.",
    )
    save(fig, "08_temporal_stride")


if __name__ == "__main__":
    main()
