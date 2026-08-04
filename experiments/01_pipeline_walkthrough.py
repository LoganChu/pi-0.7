"""What each stage of FAST actually does to a single action chunk.

Produces one six-panel figure tracing one chunk from raw actions through to
reconstructed actions, so every intermediate the pipeline builds is visible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fastlab import FastConfig, FastTokenizer, reconstruction_error
from fastlab.data import synthetic_chunks
from fastlab.viz import (
    CATEGORICAL,
    DIVERGING_CMAP,
    INK_MUTED,
    INK_SECONDARY,
    PRESENCE_CMAP,
    header,
    save,
    use_style,
)

HORIZON, ACTION_DIM, CONTROL_HZ = 50, 14, 50.0
SHOWN_DIMS = 3  # kept small so individual traces stay readable


def main() -> None:
    use_style()
    chunks = synthetic_chunks(
        n_chunks=1500, horizon=HORIZON, action_dim=ACTION_DIM,
        control_hz=CONTROL_HZ, seed=0,
    )
    tok = FastTokenizer.fit(chunks, FastConfig(scale=10.0, vocab_size=1024))
    trace = tok.encode_trace(chunks[0])
    recon = tok.decode(trace.tokens)
    err = reconstruction_error(chunks[0], recon)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    t = np.arange(HORIZON) / CONTROL_HZ

    # (1) raw actions -----------------------------------------------------
    ax = axes[0, 0]
    for d in range(SHOWN_DIMS):
        ax.plot(t, trace.raw[:, d], color=CATEGORICAL[d], label=f"dim {d}")
    ax.set_title("1 · Raw actions")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("action value")
    ax.margins(y=0.22)  # headroom so the legend never sits on the data
    ax.legend(loc="upper right", ncol=SHOWN_DIMS)
    ax.text(0.02, 0.02, "dimensions have different physical scales",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED)

    # (2) after quantile normalization ------------------------------------
    ax = axes[0, 1]
    for d in range(SHOWN_DIMS):
        ax.plot(t, trace.normalized[:, d], color=CATEGORICAL[d], label=f"dim {d}")
    ax.axhline(1.0, color=INK_MUTED, lw=1, ls="--")
    ax.axhline(-1.0, color=INK_MUTED, lw=1, ls="--")
    ax.set_title("2 · Quantile-normalized to [-1, 1]")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized value")
    ax.set_ylim(-1.35, 1.45)
    ax.legend(loc="upper right", ncol=SHOWN_DIMS)

    # (3) DCT coefficients (signed -> diverging) --------------------------
    ax = axes[1, 0]
    lim = float(np.abs(trace.coefficients).max())
    im = ax.imshow(trace.coefficients.T, aspect="auto", cmap=DIVERGING_CMAP,
                   vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title("3 · DCT-II coefficients")
    ax.set_xlabel("frequency index")
    ax.set_ylabel("action dimension")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("coefficient", color=INK_SECONDARY, fontsize=9)
    energy = (trace.coefficients[:5] ** 2).sum() / (trace.coefficients**2).sum()
    ax.text(0.98, 0.04, f"{energy:.0%} of energy in the first 5 frequencies",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK_SECONDARY)

    # (4) quantized -> sparsity -------------------------------------------
    ax = axes[1, 1]
    nonzero = (trace.quantized != 0).astype(float)
    ax.imshow(nonzero.T, aspect="auto", cmap=PRESENCE_CMAP, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(f"4 · Quantized (scale={tok.config.scale:g}) — non-zeros only")
    ax.set_xlabel("frequency index")
    ax.set_ylabel("action dimension")
    ax.grid(False)
    ax.text(0.98, 0.04, f"{trace.sparsity:.0%} of coefficients rounded to zero",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK_SECONDARY)

    # (5) flattened stream -------------------------------------------------
    ax = axes[2, 0]
    ax.plot(np.arange(trace.stream.size), trace.stream,
            color=CATEGORICAL[0], lw=1.0)
    ax.axvline(trace.trimmed.size, color=CATEGORICAL[1], lw=2)
    ax.annotate(
        f"trailing zeros dropped\n{trace.stream.size} → {trace.trimmed.size}",
        xy=(trace.trimmed.size, 0), xytext=(10, 24), textcoords="offset points",
        fontsize=9, color=INK_SECONDARY,
    )
    ax.set_title("5 · Flattened stream (frequency-major)")
    ax.set_xlabel("position in stream")
    ax.set_ylabel("quantized coefficient")

    # (6) reconstruction ---------------------------------------------------
    ax = axes[2, 1]
    ax.plot(t, trace.raw[:, 0], color=CATEGORICAL[0], label="original")
    ax.plot(t, recon[:, 0], color=CATEGORICAL[1], lw=2, ls="--", label="decoded")
    ax.set_title("6 · Round-trip (dim 0)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("action value")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.02,
            f"normalized RMSE {err.normalized_rmse:.3f} · max |err| {err.max_abs:.3f}",
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY)

    header(
        fig,
        "FAST, one chunk at a time",
        f"{HORIZON} timesteps × {ACTION_DIM} dims at {CONTROL_HZ:g} Hz  →  "
        f"{len(trace.tokens)} tokens "
        f"({HORIZON * ACTION_DIM / len(trace.tokens):.1f}× fewer than one-token-per-value)",
    )
    save(fig, "01_pipeline_walkthrough")


if __name__ == "__main__":
    main()
