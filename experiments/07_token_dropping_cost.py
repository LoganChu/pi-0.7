"""What dropping history tokens in the upper layers actually buys.

MEM's video encoder does two things to keep memory affordable. It factorizes
attention, so time and space are never attended jointly, and it drops past-frame
patch tokens once the lower layers have read them. The paper states the design;
this measures the consequence.

The interesting comparison is three-way, because factorizing and dropping are
separable decisions:

    joint       attend over all T x N tokens at once     quadratic in history
    factorized  space per frame, time per position       linear in history
    MEM         factorized, history dropped up top       linear at half the slope

MEM's win is a slope, not a ceiling: history still costs, but only in the layers
below the cut. Panel (b) shows that mechanism directly -- the timesteps entering
each layer -- rather than only its consequence.

Caveat worth carrying into any reading of panel (a): 64 patches per frame is
far fewer than MEM works with, and the joint baseline's penalty is quadratic in
exactly that number. The joint curve here is the gentlest version of itself.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fastlab.viz import CATEGORICAL, INK_MUTED, INK_SECONDARY, header, save, use_style
from memlab.cost import flops, joint_flops, measure_grid, tokens_per_layer
from memlab.encoder import EncoderConfig, MemEncoder

HISTORIES = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]
SHOWCASE = 18  # the paper's post-training history length
GIGA = 1e9


def main() -> None:
    use_style()
    mem = EncoderConfig()
    no_drop = EncoderConfig(drop_history_after=mem.depth)

    mem_g = np.array([flops(mem, t).total for t in HISTORIES]) / GIGA
    keep_g = np.array([flops(no_drop, t, drop=False).total for t in HISTORIES]) / GIGA
    joint_g = np.array([joint_flops(mem, t).total for t in HISTORIES]) / GIGA

    for t, m, k, j in zip(HISTORIES, mem_g, keep_g, joint_g):
        print(f"  T={t:>3}  MEM {m:7.2f} GF   no-drop {k:7.2f} GF   joint {j:8.2f} GF")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) the three cost regimes ------------------------------------------
    ax = axes[0]
    for values, label, slot in (
        (joint_g, "joint attention", 2),
        (keep_g, "factorized, keep history", 1),
        (mem_g, "MEM (factorized + drop)", 0),
    ):
        ax.plot(HISTORIES, values, color=CATEGORICAL[slot], marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("history length (frames)")
    ax.set_ylabel("GFLOPs per forward pass")
    ax.set_title("a · Cost of remembering")
    ax.legend(loc="upper left")
    ax.annotate(
        f"{joint_g[-1] / mem_g[-1]:.1f}× cheaper\nat T={SHOWCASE}",
        xy=(HISTORIES[-1], mem_g[-1]), xytext=(-6, -34),
        textcoords="offset points", ha="right", fontsize=9, color=INK_SECONDARY,
    )

    # (b) where the saving comes from --------------------------------------
    ax = axes[1]
    layers = np.arange(1, mem.depth + 1)
    # Drawn heavy-then-light: the two schedules are identical below the cut, and
    # letting MEM sit on top of a thicker line says that better than a legend.
    ax.step(layers, tokens_per_layer(no_drop, SHOWCASE, drop=False),
            where="mid", color=CATEGORICAL[1], lw=4.0, label="keep history")
    ax.step(layers, tokens_per_layer(mem, SHOWCASE), where="mid",
            color=CATEGORICAL[0], lw=2.0, label="MEM")
    ax.axvline(mem.drop_history_after + 0.5, color=INK_MUTED, lw=1, ls=":")
    for layer in mem.temporal_layers:
        effective = layer in mem.effective_temporal_layers
        ax.plot(layer + 1, -1.6, marker="^", ms=7, clip_on=False,
                color=CATEGORICAL[0] if effective else INK_MUTED)
    ax.set_ylim(-3.0, SHOWCASE + 1.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"timesteps entering layer (T={SHOWCASE})")
    ax.set_title("b · Why: history stops propagating")
    ax.legend(loc="center left")
    ax.annotate(
        f"▲ temporal layer — grey has no history left to mix\n"
        f"    {len(mem.temporal_layers)} nominal, "
        f"{len(mem.effective_temporal_layers)} effective",
        xy=(0.0, -0.235), xycoords="axes fraction",
        fontsize=8.5, color=INK_MUTED,
    )

    # (c) does a real machine agree ----------------------------------------
    ax = axes[2]
    probe = [1, 6, 12, 18]
    timings = measure_grid(
        [(f"mem{t}", mem, t) for t in probe]
        + [(f"keep{t}", no_drop, t) for t in probe]
    )
    mem_s = [timings[f"mem{t}"] * 1e3 for t in probe]
    keep_s = [timings[f"keep{t}"] * 1e3 for t in probe]
    ax.plot(probe, keep_s, color=CATEGORICAL[1], marker="s", label="keep history")
    ax.plot(probe, mem_s, color=CATEGORICAL[0], marker="o", label="MEM")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("history length (frames)")
    ax.set_ylabel("CPU forward pass (ms per sample)")
    ax.set_title("c · Measured, not just counted")
    ax.legend(loc="upper left")
    print(f"  measured MEM {mem_s[0]:.0f} -> {mem_s[-1]:.0f} ms;  "
          f"predicted {mem_g[-1] / mem_g[0]:.1f}x, "
          f"measured {mem_s[-1] / mem_s[0]:.1f}x")

    shared = MemEncoder(mem).n_params()
    separate = MemEncoder(EncoderConfig(share_time_space_weights=False)).n_params()
    header(
        fig,
        "MEM buys history cheaply by not carrying it all the way up",
        f"{SHOWCASE} frames cost {mem_g[-1] / mem_g[0]:.1f}× a single frame, against "
        f"{joint_g[-1] / joint_g[0]:.0f}× for joint attention. Sharing the spatial "
        f"attention weights along time keeps the encoder at {shared / 1e6:.2f}M "
        f"parameters — a separately parameterized temporal block would need "
        f"{separate / 1e6:.2f}M.",
    )
    save(fig, "07_token_dropping_cost")


if __name__ == "__main__":
    main()
