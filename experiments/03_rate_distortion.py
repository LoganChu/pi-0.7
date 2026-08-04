"""The `scale` knob: what does each extra token buy in reconstruction error?

Sweeping the quantization scale traces FAST's rate-distortion curve. The naive
per-timestep binning baseline gets the same treatment (sweeping bin count), and
the contrast is the point: naive tokenization can only move along the *error*
axis, because its token count is pinned at T x D no matter what.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fastlab import (
    FastConfig,
    FastTokenizer,
    NaiveBinningTokenizer,
    reconstruction_error,
)
from fastlab.data import synthetic_chunks
from fastlab.viz import CATEGORICAL, INK_MUTED, INK_SECONDARY, header, save, use_style

HORIZON, ACTION_DIM = 50, 14
SCALES = [1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 25.0, 40.0, 70.0, 120.0]
BIN_COUNTS = [8, 16, 32, 64, 128, 256, 512, 1024]
VOCAB = 1024
N_EVAL = 300


def main() -> None:
    use_style()
    chunks = synthetic_chunks(
        n_chunks=1500, horizon=HORIZON, action_dim=ACTION_DIM, control_hz=50.0, seed=0
    )
    eval_chunks = chunks[:N_EVAL]
    naive_tokens = HORIZON * ACTION_DIM

    fast_rows = []
    infeasible = []
    for scale in SCALES:
        # Past some scale the quantized coefficients span more distinct integers
        # than the vocabulary has room for, leaving no budget for merges. That
        # ceiling is a real constraint on how far the accuracy knob can turn.
        try:
            tok = FastTokenizer.fit(chunks, FastConfig(scale=scale, vocab_size=VOCAB))
        except ValueError as exc:
            infeasible.append(scale)
            print(f"  scale={scale:<6g} infeasible at vocab={VOCAB}: {exc}")
            continue
        toks = tok.encode_batch(eval_chunks)
        recon = tok.decode_batch(toks)
        fast_rows.append(
            (
                scale,
                float(np.mean([len(t) for t in toks])),
                reconstruction_error(eval_chunks, recon).normalized_rmse,
            )
        )
        print(f"  scale={scale:<6g} tokens={fast_rows[-1][1]:6.1f} "
              f"nrmse={fast_rows[-1][2]:.4f}")

    naive_rows = []
    for n_bins in BIN_COUNTS:
        nb = NaiveBinningTokenizer.fit(chunks, n_bins=n_bins)
        recon = np.stack(
            [nb.decode(nb.encode(c), HORIZON, ACTION_DIM) for c in eval_chunks]
        )
        naive_rows.append(
            (n_bins, naive_tokens, reconstruction_error(eval_chunks, recon).normalized_rmse)
        )

    fast = np.array([(r[1], r[2]) for r in fast_rows])
    naive = np.array([(r[1], r[2]) for r in naive_rows])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) the rate-distortion plane ---------------------------------------
    ax = axes[0]
    ax.plot(fast[:, 0], fast[:, 1], color=CATEGORICAL[0], marker="o",
            label="FAST (sweeping scale)")
    ax.plot(naive[:, 0], naive[:, 1], color=CATEGORICAL[1], marker="s",
            ls="none", label="naive binning (sweeping bins)")
    ax.set_yscale("log")
    ax.set_xlabel("mean tokens per chunk  →  cost")
    ax.set_ylabel("normalized RMSE  →  distortion")
    ax.set_title("a · Rate–distortion")
    ax.legend(loc="lower left")
    ax.grid(True, axis="both")

    for scale, tokens, err in [fast_rows[0], fast_rows[len(fast_rows) // 2], fast_rows[-1]]:
        ax.annotate(f"scale={scale:g}", xy=(tokens, err), xytext=(8, 10),
                    textcoords="offset points", fontsize=9, color=INK_SECONDARY)
    ax.axvline(naive_tokens, color=INK_MUTED, lw=1, ls="--")
    ax.text(naive_tokens - 14, fast[:, 1].min(), f"naive is pinned at {naive_tokens} tokens",
            rotation=90, ha="right", va="bottom", fontsize=9, color=INK_MUTED)
    if infeasible:
        ax.text(0.40, 0.93,
                f"scale ≥ {min(infeasible):g} needs a vocabulary larger than {VOCAB}\n"
                "(the integer alphabet outgrows it,\nleaving no room for merges)",
                transform=ax.transAxes, va="top", fontsize=9, color=INK_SECONDARY)

    # (b) matched-error comparison ----------------------------------------
    # Read off how many tokens FAST needs to hit each naive configuration's
    # error. Interpolation is on log-error, which is where the curve is smooth.
    ax = axes[1]
    order = np.argsort(fast[:, 1])
    matched = np.interp(
        np.log(naive[:, 1]), np.log(fast[order, 1]), fast[order, 0]
    )
    keep = (naive[:, 1] >= fast[:, 1].min()) & (naive[:, 1] <= fast[:, 1].max())

    x = np.arange(int(keep.sum()))
    width = 0.38
    ax.bar(x - width / 2, np.full(keep.sum(), naive_tokens), width,
           color=CATEGORICAL[1], label="naive binning")
    ax.bar(x + width / 2, matched[keep], width,
           color=CATEGORICAL[0], label="FAST at equal error")
    for xi, m in zip(x, matched[keep]):
        ax.text(xi + width / 2, m, f"{naive_tokens/m:.1f}×", ha="center",
                va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x, [f"{b}\nbins" for b, k in zip(BIN_COUNTS, keep) if k])
    ax.set_ylabel("tokens per chunk")
    ax.set_title("b · Tokens needed for the same reconstruction error")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.6)

    header(
        fig,
        "What the quantization scale buys",
        f"{HORIZON} steps × {ACTION_DIM} dims at 50 Hz. FAST trades tokens for "
        "accuracy along a curve; naive binning has no such knob.",
    )
    save(fig, "03_rate_distortion")


if __name__ == "__main__":
    main()
