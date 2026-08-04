"""What did BPE actually learn to merge?

A BPE vocabulary that stayed mostly length-1 learned nothing: all the
compression would then be coming from zero-trimming alone. This measures how
much each stage contributes, what the learned tokens look like, and where extra
vocabulary stops paying for itself.
"""

from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np

from fastlab import FastConfig, FastTokenizer
from fastlab.dct import flatten, forward_dct, quantize
from fastlab.data import synthetic_chunks
from fastlab.normalize import apply as apply_norm
from fastlab.normalize import fit as fit_norm
from fastlab.viz import CATEGORICAL, INK_SECONDARY, header, save, use_style

HORIZON, ACTION_DIM = 50, 14
VOCAB_SIZES = [512, 1024, 2048, 4096, 8192]
REFERENCE_VOCAB = 1024
N_EVAL = 300


def main() -> None:
    use_style()
    chunks = synthetic_chunks(
        n_chunks=1500, horizon=HORIZON, action_dim=ACTION_DIM, control_hz=50.0, seed=0
    )
    eval_chunks = chunks[:N_EVAL]

    tok = FastTokenizer.fit(
        chunks, FastConfig(scale=10.0, vocab_size=REFERENCE_VOCAB)
    )
    traces = [tok.encode_trace(c) for c in eval_chunks]

    # Stage-by-stage token budget.
    stats = fit_norm(chunks)
    normalized = apply_norm(eval_chunks, stats)
    raw_len = HORIZON * ACTION_DIM
    nonzero_len = float(
        np.mean([
            np.count_nonzero(flatten(quantize(forward_dct(c), 10.0), "freq_major"))
            for c in normalized
        ])
    )
    trimmed_len = float(np.mean([t.trimmed.size for t in traces]))
    token_len = float(np.mean([len(t.tokens) for t in traces]))

    # Usage-weighted merge lengths: how many coefficients an emitted token
    # actually stands for, in practice rather than in the vocabulary.
    lengths = tok.bpe.merge_lengths()
    usage = collections.Counter()
    for t in traces:
        usage.update(t.tokens)
    used_lengths = np.array(
        [lengths.get(i, 1) for i, c in usage.items() for _ in range(c)]
    )
    vocab_lengths = np.array(list(lengths.values()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) where the compression comes from --------------------------------
    ax = axes[0]
    labels = ["raw\nT×D", "after\nzero-trim", "after\nBPE"]
    values = [raw_len, trimmed_len, token_len]
    ax.bar(range(3), values, color=CATEGORICAL[0], width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10,
                color=INK_SECONDARY)
    # Reference, not a stage: the count you would get by transmitting only the
    # non-zero coefficients. BPE lands *below* it, because a merged token can
    # stand for several non-zero coefficients at once.
    ax.axhline(nonzero_len, color=CATEGORICAL[1], lw=2, ls="--")
    ax.text(2.45, nonzero_len, f"non-zero coefficients ({nonzero_len:.0f})",
            ha="right", va="bottom", fontsize=9, color=CATEGORICAL[1])
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("symbols per chunk")
    ax.set_title("a · Where the compression comes from")

    # (b) merge-length distribution ---------------------------------------
    ax = axes[1]
    max_len = int(max(vocab_lengths.max(), used_lengths.max()))
    bins = np.arange(0.5, max_len + 1.5)
    ax.hist(vocab_lengths, bins=bins, density=True, histtype="step", lw=2,
            color=CATEGORICAL[0], label="vocabulary")
    ax.hist(used_lengths, bins=bins, density=True, histtype="step", lw=2,
            color=CATEGORICAL[1], label="usage-weighted")
    ax.set_xlabel("coefficients merged into one token")
    ax.set_ylabel("fraction")
    ax.set_title("b · What BPE learned to merge")
    # The tail runs to ~60 (long zero runs) but is vanishingly thin; clipping
    # keeps the mass where the reader can actually see it.
    ax.set_xlim(0.5, 16.5)
    ax.legend(loc="upper right")
    ax.text(0.98, 0.55,
            f"mean emitted token covers {used_lengths.mean():.2f} coefficients\n"
            f"longest merge in vocab: {max_len} (a zero run)",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK_SECONDARY)

    # (c) vocabulary size sweep -------------------------------------------
    ax = axes[2]
    sizes, means = [], []
    for v in VOCAB_SIZES:
        t = FastTokenizer.fit(chunks, FastConfig(scale=10.0, vocab_size=v))
        toks = t.encode_batch(eval_chunks)
        sizes.append(v)
        means.append(float(np.mean([len(x) for x in toks])))
        print(f"  vocab={v:<6} tokens={means[-1]:.1f}")
    ax.plot(sizes, means, color=CATEGORICAL[0], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(VOCAB_SIZES, [str(v) for v in VOCAB_SIZES])
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("mean tokens per chunk")
    ax.set_title("c · Vocabulary size vs token cost")
    ax.annotate(f"{means[0]:.0f}", xy=(sizes[0], means[0]), xytext=(6, 6),
                textcoords="offset points", fontsize=9, color=INK_SECONDARY)
    ax.annotate(f"{means[-1]:.0f}", xy=(sizes[-1], means[-1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", fontsize=9,
                color=INK_SECONDARY)
    per_doubling = 1.0 - (means[-1] / means[0]) ** (
        1.0 / np.log2(sizes[-1] / sizes[0])
    )
    ax.text(0.5, 0.9, f"each doubling of the vocabulary\nbuys ≈{per_doubling:.0%} fewer tokens",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color=INK_SECONDARY)

    header(
        fig,
        "BPE anatomy",
        f"Zero-trimming does {raw_len/trimmed_len:.1f}× of the work; BPE adds "
        f"another {trimmed_len/token_len:.2f}× on top, for "
        f"{raw_len/token_len:.1f}× overall.",
    )
    save(fig, "05_bpe_anatomy")


if __name__ == "__main__":
    main()
