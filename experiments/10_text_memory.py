"""Does compressed text memory stay small, and does it hold the right things?

MEM's argument for storing the long horizon as language is that language is what
stays small when you summarize it. That is two claims pulling against each
other -- a memory that stays small by forgetting is not a memory -- so both get
measured:

    (a) growth      tokens against events absorbed, over whole episodes.
    (b) redundancy  compaction is a *merge*, so what it buys depends entirely on
                    how much the event stream repeats. Sweeping the number of
                    identical items in one cabinet isolates that -- it is the
                    paper's own "three bowls" example, turned into an axis.
    (c) budget      sweep the token ceiling and watch what compaction gives up
                    first. The drop order is a policy decision, not an accident.
    (d) recall      the timeline from experiment 09 with a third state: a fact
                    is on screen, or only in text memory, or lost.

Panel (d) is where MEM's two halves meet -- experiment 09 showed which facts
leave the observation, and this shows the text memory catching them.

Sizes are counted with a real tokenizer (`memory_tokenizer.py`), not words,
because tokens are what the memory costs a policy's context.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from fastlab.viz import (
    BASELINE,
    CATEGORICAL,
    INK_MUTED,
    INK_SECONDARY,
    header,
    save,
    use_style,
)
from memlab.memory_tokenizer import MemoryTokenizer
from memlab.oracle import collect
from memlab.render import cell_on_screen
from memlab.text_memory import MemoryConfig, memory_trace
from memlab.world import default_layout

N_EPISODES = 24
CLUTTER_SWEEP = [0, 1, 2, 3, 4, 5, 6]
BUDGETS = [6, 8, 10, 12, 16, 20, 24, 32]
TASKS = ("find_object", "tidy_up")
SHOWCASE = ("tidy_up", 0)


def main() -> None:
    use_style()
    tokenizer = MemoryTokenizer.from_episodes(n_episodes=8)
    compact = replace(MemoryConfig(), count_tokens=tokenizer.count)
    verbatim = replace(compact, compact=False)
    print(f"  memory vocabulary: {tokenizer.vocab_size} tokens")

    episodes = {t: collect(t, N_EPISODES, seed=1, keep_frames=False) for t in TASKS}

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.4))
    _panel_growth(axes[0, 0], episodes, compact, verbatim)
    merged = _panel_redundancy(axes[0, 1], compact, verbatim, tokenizer)
    _panel_budget(axes[1, 0], episodes["tidy_up"], tokenizer)
    coverage = _panel_recall(axes[1, 1], compact)

    header(
        fig,
        "Compaction is a merge, so what it saves depends on what repeats",
        f"Adding identical items costs the transcript {merged[0]:.1f} tokens each and "
        f"the compacted memory {merged[1]:.1f}. Compaction does not bound the memory — "
        f"the budget does — but at 48 tokens it still holds {coverage:.0%} of the "
        f"facts that have left the screen.",
    )
    save(fig, "10_text_memory")


def _growth(episodes, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Mean memory size at each count of events absorbed."""
    xs, ys = [], []
    for episode in episodes:
        trace = memory_trace(episode, cfg)
        absorbed = np.searchsorted(
            [e.step for e in episode.events], np.arange(len(episode)), side="right"
        )
        xs.extend(absorbed)
        ys.extend(trace.tokens)
    xs, ys = np.array(xs), np.array(ys)
    # Two events can land on the same step (open, then see), so some
    # absorbed-counts never appear. Keep only the ones with samples.
    counts = np.unique(xs)
    return counts, np.array([ys[xs == n].mean() for n in counts])


def _panel_growth(ax, episodes, compact, verbatim) -> None:
    """Memory size against events absorbed, over whole episodes."""
    for slot, (task, eps) in enumerate(episodes.items()):
        for cfg, style, kind in ((verbatim, "--", "verbatim"), (compact, "-", "compacted")):
            counts, means = _growth(eps, cfg)
            ax.plot(counts, means, style, color=CATEGORICAL[slot],
                    label=f"{task}, {kind}")
    ax.set_xlabel("events absorbed")
    ax.set_ylabel("memory size (tokens)")
    ax.set_title("a · Growth over an episode")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.annotate("compaction bends the curve but does not flatten it:\n"
                "every event here is a genuinely new fact",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=8.5, color=INK_MUTED)


def _panel_redundancy(ax, compact, verbatim, tokenizer) -> tuple[float, float]:
    """Final memory size as identical items pile into one cabinet.

    This isolates the mechanism the paper names: three separately-observed
    bowls should cost what one costs, plus a number word.
    """
    curves = {}
    for cfg, kind in ((verbatim, "verbatim"), (compact, "compacted")):
        sizes = []
        for n in CLUTTER_SWEEP:
            eps = collect("find_object", N_EPISODES, seed=1,
                          keep_frames=False, clutter=n)
            sizes.append(np.mean([memory_trace(e, cfg).tokens[-1] for e in eps]))
        curves[kind] = np.array(sizes)

    slopes = []
    for slot, (kind, sizes) in enumerate(curves.items()):
        style = "--" if kind == "verbatim" else "-"
        ax.plot(CLUTTER_SWEEP, sizes, style, color=CATEGORICAL[slot],
                marker="o", label=kind)
        slope = float(np.polyfit(CLUTTER_SWEEP, sizes, 1)[0])
        slopes.append(slope)
        ax.annotate(f"+{slope:.1f} tok/item", xy=(CLUTTER_SWEEP[-1], sizes[-1]),
                    xytext=(-4, 8), textcoords="offset points", ha="right",
                    fontsize=8.5, color=CATEGORICAL[slot])
    ax.set_xlabel("identical items in one cabinet")
    ax.set_ylabel("memory size at episode end (tokens)")
    ax.set_title("b · The paper's compression example, swept")
    ax.legend(loc="center left", fontsize=8.5)
    ax.annotate("“saw mug. saw mug. saw mug.”\ncollapses to “three mugs in cabinet 1”",
                xy=(0.04, 0.58), xycoords="axes fraction",
                fontsize=8.5, color=INK_MUTED)
    return slopes[0], slopes[1]


def _panel_budget(ax, episodes, tokenizer) -> None:
    """What compaction gives up as the ceiling tightens."""
    kept_searched, kept_objects = [], []
    for budget in BUDGETS:
        # protected=() on purpose: exempting the instruction's object would
        # hide the drop order, which is the thing being shown.
        cfg = replace(MemoryConfig(), count_tokens=tokenizer.count,
                      max_tokens=budget, protected=())
        finals = [memory_trace(e, cfg).beliefs[-1] for e in episodes]
        kept_searched.append(np.mean([len(b.searched) for b in finals]))
        kept_objects.append(np.mean([len(b.placements) for b in finals]))

    ax.plot(BUDGETS, kept_searched, color=CATEGORICAL[0], marker="o",
            label="cabinets known searched")
    ax.plot(BUDGETS, kept_objects, color=CATEGORICAL[1], marker="s",
            label="object locations held")
    ax.set_ylim(-0.15, 3.4)
    ax.set_xlabel("token budget")
    ax.set_ylabel("facts retained at episode end")
    ax.set_title("c · What it gives up first")
    ax.legend(loc="center right", fontsize=8.5)
    ax.annotate("object locations go first: re-searching a\n"
                "cabinet costs more than re-finding one object",
                xy=(0.03, 0.42), xycoords="axes fraction",
                fontsize=8.5, color=INK_MUTED)


def _panel_recall(ax, cfg) -> float:
    """On screen, only in memory, or lost -- for one showcase episode."""
    task, seed = SHOWCASE
    episode = collect(task, 1, seed=seed, keep_frames=False)[0]
    trace = memory_trace(episode, cfg)
    layout = default_layout()
    clock = np.arange(len(episode))

    rows, covered, uncovered = [], 0, 0
    for step, text in episode.event_steps():
        if not text.startswith("opened cabinet"):
            continue
        cid = int(text.rsplit(" ", 1)[1])
        established = clock >= step
        on_screen = cell_on_screen(episode.positions, layout.cabinets[cid]) & established
        in_memory = np.array([b.knows_searched(cid) for b in trace.beliefs]) & established
        rows.append((f"cabinet {cid}", on_screen, in_memory & ~on_screen))
        covered += int((in_memory & ~on_screen).sum())
        uncovered += int((established & ~on_screen).sum())

    for i, (label, on_screen, only_memory) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.broken_barh(_spans(only_memory), (y - 0.22, 0.44),
                       facecolors=CATEGORICAL[0], alpha=0.30, edgecolors="none")
        ax.broken_barh(_spans(on_screen), (y - 0.22, 0.44),
                       facecolors=CATEGORICAL[0], edgecolors="none")
        ax.axhline(y, color=BASELINE, lw=0.8, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, len(episode))
    ax.set_xlabel("step        solid = on screen,  pale = only in text memory")
    ax.set_title(f"d · “already searched here”, {task}")
    ax.grid(False)
    return covered / uncovered if uncovered else 1.0


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


if __name__ == "__main__":
    main()
