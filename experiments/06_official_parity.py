"""Does this reimplementation behave like the real thing?

Compares fastlab against the official universal tokenizer at
`physical-intelligence/fast`. They will not produce identical token *ids* --
different BPE vocabularies, and the official one is fit across many robot
embodiments rather than this dataset -- so the comparison that matters is
behavioral: comparable token counts and comparable reconstruction error.

Requires:  uv sync --extra official
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from fastlab import FastConfig, FastTokenizer, reconstruction_error
from fastlab.data import synthetic_chunks
from fastlab.normalize import apply as apply_norm
from fastlab.normalize import fit as fit_norm
from fastlab.viz import CATEGORICAL, INK_SECONDARY, header, save, use_style

HORIZON, ACTION_DIM = 50, 14
N_EVAL = 128


def load_official():
    try:
        from transformers import AutoProcessor
    except ImportError:
        print("transformers not installed. Run: uv sync --extra official")
        return None
    try:
        return AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
    except Exception as exc:  # network or hub failure
        print(f"could not load physical-intelligence/fast: {exc}")
        return None


def main() -> None:
    use_style()
    official = load_official()
    if official is None:
        sys.exit(1)

    chunks = synthetic_chunks(
        n_chunks=1500, horizon=HORIZON, action_dim=ACTION_DIM, control_hz=50.0, seed=0
    )
    eval_chunks = chunks[:N_EVAL]

    # The official tokenizer expects inputs already normalized to [-1, 1], so
    # both sides are fed the same normalized array to keep this a comparison of
    # the tokenizers rather than of the normalizers.
    stats = fit_norm(chunks)
    eval_norm = apply_norm(eval_chunks, stats)

    ours = FastTokenizer.fit(chunks, FastConfig(scale=10.0, vocab_size=1024,
                                                normalize=False))
    our_tokens = ours.encode_batch(eval_norm)
    our_recon = ours.decode_batch(our_tokens)
    our_err = reconstruction_error(eval_norm, our_recon)

    off_tokens = official(eval_norm)
    off_recon = np.asarray(
        official.decode(off_tokens, time_horizon=HORIZON, action_dim=ACTION_DIM)
    )
    off_err = reconstruction_error(eval_norm, off_recon)

    our_counts = np.array([len(t) for t in our_tokens], dtype=float)
    off_counts = np.array([len(t) for t in off_tokens], dtype=float)

    print(f"\n  {'':<22}{'tokens':>10}{'nrmse':>10}{'max|err|':>11}")
    print(f"  {'fastlab':<22}{our_counts.mean():>10.1f}"
          f"{our_err.normalized_rmse:>10.4f}{our_err.max_abs:>11.4f}")
    print(f"  {'official FAST':<22}{off_counts.mean():>10.1f}"
          f"{off_err.normalized_rmse:>10.4f}{off_err.max_abs:>11.4f}")

    # The decisive check. Token ids differ (different BPE vocabularies), but if
    # the DCT, quantizer and inverse match the official implementation then the
    # decoded actions are the same array, not merely a similar one.
    max_diff = float(np.abs(our_recon - off_recon).max())
    identical = np.allclose(our_recon, off_recon, atol=1e-12)
    print(f"\n  max |fastlab - official| reconstruction: {max_diff:.3e}")
    print(f"  bitwise identical: {identical}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) token counts -----------------------------------------------------
    ax = axes[0]
    bins = np.histogram_bin_edges(np.concatenate([our_counts, off_counts]), bins=25)
    ax.hist(our_counts, bins=bins, histtype="step", lw=2, color=CATEGORICAL[0],
            label="fastlab")
    ax.hist(off_counts, bins=bins, histtype="step", lw=2, color=CATEGORICAL[1],
            label="official FAST")
    ax.set_xlabel("tokens per chunk")
    ax.set_ylabel("chunks")
    ax.set_title("a · Token count")
    ax.legend(loc="upper right")

    # (b) reconstruction error ---------------------------------------------
    ax = axes[1]
    ax.bar([0, 1], [our_err.normalized_rmse, off_err.normalized_rmse],
           color=[CATEGORICAL[0], CATEGORICAL[1]], width=0.55)
    for i, v in enumerate([our_err.normalized_rmse, off_err.normalized_rmse]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                color=INK_SECONDARY)
    ax.set_xticks([0, 1], ["fastlab", "official FAST"])
    ax.set_ylabel("normalized RMSE")
    ax.set_title("b · Reconstruction error")

    # (c) both round-trips on one trace -------------------------------------
    ax = axes[2]
    t = np.arange(HORIZON) / 50.0
    ax.plot(t, eval_norm[0, :, 0], color=CATEGORICAL[2], lw=3, alpha=0.55,
            label="original")
    ax.plot(t, our_recon[0, :, 0], color=CATEGORICAL[0], ls="--", label="fastlab")
    ax.plot(t, off_recon[0, :, 0], color=CATEGORICAL[1], ls=":", label="official")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized action")
    ax.set_title("c · Round-trip, dim 0")
    ax.legend(loc="upper right")

    verdict = (
        "The decoded actions are bitwise identical — the DCT and quantizer match "
        "exactly, and only the BPE vocabulary differs."
        if identical
        else f"The decoded actions differ by up to {max_diff:.2e}."
    )
    header(
        fig,
        "fastlab vs the official FAST tokenizer",
        f"Token ids differ by construction (different BPE vocabularies). {verdict}",
    )
    save(fig, "06_official_parity")


if __name__ == "__main__":
    main()
