"""Does frequency-major vs dimension-major flattening actually matter?

BPE can only merge *adjacent* symbols, so how the (T, D) coefficient matrix is
serialized decides where the zeros end up and therefore how much BPE can
compress. This was an open question on the openpi tracker (issue #306).

    freq_major: [f0d0 f0d1 ... f0dD, f1d0 ...]  -> zeros pool in one tail run
    dim_major:  [d0f0 d0f1 ... d0fT, d1f0 ...]  -> D shorter zero runs
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fastlab import FastConfig, FastTokenizer, reconstruction_error
from fastlab.dct import drop_trailing_zeros, flatten, forward_dct, quantize
from fastlab.data import synthetic_chunks
from fastlab.normalize import apply as apply_norm
from fastlab.normalize import fit as fit_norm
from fastlab.viz import CATEGORICAL, INK_SECONDARY, header, save, use_style

HORIZON, ACTION_DIM = 50, 14
ORDERS = ["freq_major", "dim_major"]
N_EVAL = 300


def main() -> None:
    use_style()
    chunks = synthetic_chunks(
        n_chunks=1500, horizon=HORIZON, action_dim=ACTION_DIM, control_hz=50.0, seed=0
    )
    stats = fit_norm(chunks)
    normalized = apply_norm(chunks, stats)

    results = {}
    for order in ORDERS:
        cfg = FastConfig(scale=10.0, vocab_size=1024, flatten_order=order)
        tok = FastTokenizer.fit(chunks, cfg)
        toks = tok.encode_batch(chunks[:N_EVAL])
        recon = tok.decode_batch(toks)

        # Pre-BPE stream length isolates what the zero-run structure alone buys,
        # before BPE gets a chance to merge anything.
        trimmed = [
            drop_trailing_zeros(
                flatten(quantize(forward_dct(c), cfg.scale), order)
            ).size
            for c in normalized[:N_EVAL]
        ]
        results[order] = {
            "tokens": np.array([len(t) for t in toks], dtype=float),
            "trimmed": np.array(trimmed, dtype=float),
            "err": reconstruction_error(chunks[:N_EVAL], recon).normalized_rmse,
        }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) stream length before BPE ---------------------------------------
    ax = axes[0]
    ax.bar(
        [0, 1],
        [results[o]["trimmed"].mean() for o in ORDERS],
        color=[CATEGORICAL[0], CATEGORICAL[1]], width=0.55,
    )
    for i, o in enumerate(ORDERS):
        v = results[o]["trimmed"].mean()
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10,
                color=INK_SECONDARY)
    ax.set_xticks([0, 1], ["frequency-major", "dimension-major"])
    ax.set_title("a · Stream length after zero-trim")
    ax.set_ylabel("coefficients (pre-BPE)")
    ax.axhline(HORIZON * ACTION_DIM, color=INK_SECONDARY, lw=1, ls="--")
    ax.text(1.4, HORIZON * ACTION_DIM, f"untrimmed = {HORIZON*ACTION_DIM}",
            ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)

    # (b) final token count ------------------------------------------------
    ax = axes[1]
    ax.bar(
        [0, 1],
        [results[o]["tokens"].mean() for o in ORDERS],
        color=[CATEGORICAL[0], CATEGORICAL[1]], width=0.55,
    )
    for i, o in enumerate(ORDERS):
        v = results[o]["tokens"].mean()
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10,
                color=INK_SECONDARY)
    ax.set_xticks([0, 1], ["frequency-major", "dimension-major"])
    ax.set_title("b · Tokens after BPE")
    ax.set_ylabel("mean tokens per chunk")

    # (c) distribution -----------------------------------------------------
    ax = axes[2]
    bins = np.histogram_bin_edges(
        np.concatenate([results[o]["tokens"] for o in ORDERS]), bins=30
    )
    for i, o in enumerate(ORDERS):
        ax.hist(results[o]["tokens"], bins=bins, histtype="step", lw=2,
                color=CATEGORICAL[i], label=o.replace("_", "-"))
    ax.set_title("c · Token-count distribution")
    ax.set_xlabel("tokens per chunk")
    ax.set_ylabel("chunks")
    ax.legend(loc="upper right")

    best, worst = sorted(ORDERS, key=lambda o: results[o]["tokens"].mean())
    delta = results[worst]["tokens"].mean() / results[best]["tokens"].mean()
    shorter = min(ORDERS, key=lambda o: results[o]["trimmed"].mean())
    trim_gap = results[worst]["trimmed"].mean() / results[shorter]["trimmed"].mean()

    if shorter != best:
        # The interesting case: whichever order trims shorter loses anyway,
        # because BPE merges the other one's repeated per-dimension zero runs.
        story = (
            f"{shorter.replace('_','-')} trims {trim_gap:.1f}× shorter before BPE — "
            f"yet {best.replace('_','-')} still ends up {delta:.2f}× cheaper, "
            "because BPE merges its repeated zero runs."
        )
    else:
        story = f"{best.replace('_','-')} wins at both stages, {delta:.2f}× fewer tokens."
    header(
        fig,
        "Flatten order: frequency-major vs dimension-major",
        f"Reconstruction error is identical either way (the DCT and quantizer are "
        f"untouched). {story}",
    )
    save(fig, "02_flatten_order")

    print(f"\n  {'order':<18}{'trimmed':>10}{'tokens':>10}{'nrmse':>10}")
    for o in ORDERS:
        r = results[o]
        print(f"  {o:<18}{r['trimmed'].mean():>10.1f}{r['tokens'].mean():>10.1f}"
              f"{r['err']:>10.4f}")


if __name__ == "__main__":
    main()
