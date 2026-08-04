"""Why FAST's advantage grows with control frequency.

This is the FAST paper's central claim. A fixed-length action chunk sampled at a
higher control rate spans less physical motion, so consecutive actions are more
correlated and the trajectory is smoother in frequency space. Naive per-timestep
binning cannot exploit that at all -- it always spends T x D tokens -- while
FAST's cost falls as the signal gets smoother.

The underlying motion bandwidth is held fixed; only the sampling rate moves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fastlab import FastConfig, FastTokenizer, reconstruction_error
from fastlab.data import synthetic_chunks
from fastlab.viz import CATEGORICAL, INK_SECONDARY, header, save, use_style

HORIZON, ACTION_DIM = 50, 14
CONTROL_RATES = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]
BANDWIDTH_HZ = 2.0
N_EVAL = 300


def main() -> None:
    use_style()
    naive_tokens = HORIZON * ACTION_DIM
    rows = []

    for hz in CONTROL_RATES:
        chunks = synthetic_chunks(
            n_chunks=1200, horizon=HORIZON, action_dim=ACTION_DIM,
            control_hz=hz, bandwidth_hz=BANDWIDTH_HZ, seed=0,
        )
        tok = FastTokenizer.fit(chunks, FastConfig(scale=10.0, vocab_size=1024))
        toks = tok.encode_batch(chunks[:N_EVAL])
        recon = tok.decode_batch(toks)
        traces = [tok.encode_trace(c) for c in chunks[:64]]
        rows.append(
            {
                "hz": hz,
                "tokens": float(np.mean([len(t) for t in toks])),
                "sparsity": float(np.mean([tr.sparsity for tr in traces])),
                "nrmse": reconstruction_error(chunks[:N_EVAL], recon).normalized_rmse,
            }
        )
        print(f"  {hz:>6g} Hz  tokens={rows[-1]['tokens']:6.1f}  "
              f"sparsity={rows[-1]['sparsity']:.3f}  nrmse={rows[-1]['nrmse']:.4f}")

    hz = np.array([r["hz"] for r in rows])
    tokens = np.array([r["tokens"] for r in rows])
    sparsity = np.array([r["sparsity"] for r in rows])
    ratio = naive_tokens / tokens

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) absolute token cost ---------------------------------------------
    ax = axes[0]
    ax.plot(hz, tokens, color=CATEGORICAL[0], marker="o", label="FAST")
    ax.plot(hz, np.full_like(hz, naive_tokens), color=CATEGORICAL[1],
            marker="s", label="naive binning")
    ax.set_xscale("log")
    ax.set_xticks(CONTROL_RATES, [f"{h:g}" for h in CONTROL_RATES])
    ax.set_xlabel("control frequency (Hz)")
    ax.set_ylabel("mean tokens per chunk")
    ax.set_title("a · Token cost per chunk")
    ax.legend(loc="center left")

    # (b) the compression ratio -------------------------------------------
    ax = axes[1]
    ax.plot(hz, ratio, color=CATEGORICAL[0], marker="o")
    ax.set_xscale("log")
    ax.set_xticks(CONTROL_RATES, [f"{h:g}" for h in CONTROL_RATES])
    ax.set_xlabel("control frequency (Hz)")
    ax.set_ylabel("naive tokens ÷ FAST tokens")
    ax.set_title("b · Compression advantage")
    for i in (0, len(hz) - 1):
        ax.annotate(f"{ratio[i]:.1f}×", xy=(hz[i], ratio[i]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=INK_SECONDARY)

    # (c) the mechanism ----------------------------------------------------
    ax = axes[2]
    ax.plot(hz, sparsity * 100, color=CATEGORICAL[0], marker="o")
    ax.set_xscale("log")
    ax.set_xticks(CONTROL_RATES, [f"{h:g}" for h in CONTROL_RATES])
    ax.set_xlabel("control frequency (Hz)")
    ax.set_ylabel("coefficients rounded to zero (%)")
    ax.set_title("c · Why: the spectrum gets sparser")
    ax.set_ylim(0, 100)

    header(
        fig,
        "FAST wins by more as control frequency rises",
        f"Motion bandwidth fixed at {BANDWIDTH_HZ:g} Hz; only the sampling rate "
        f"changes. Advantage goes {ratio[0]:.1f}× → {ratio[-1]:.1f}× "
        f"from {hz[0]:g} Hz to {hz[-1]:g} Hz.",
    )
    save(fig, "04_control_frequency")


if __name__ == "__main__":
    main()
