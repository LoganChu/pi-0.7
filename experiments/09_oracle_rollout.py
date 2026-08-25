"""One episode, end to end: where the agent went, what it saw, what it must recall.

The first two panels are just *looking at the thing* -- the oracle's path across
the kitchen, and the 64x64 crops it actually received at each moment something
happened. Worth having before trusting any number computed from these episodes.

Panel (c) is the one that pays for itself. For each fact a policy would need, it
plots whether that fact is visible in the agent's own view, from the moment the
fact is established onward. Blue means the observation still records it; blank
means it survives only in memory.

It also surfaces an uncomfortable result about `find_object` -- see the README.

Companion animation, same episode:

    python -m memlab.render --task find_object --seed 0
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from fastlab.viz import (
    BASELINE,
    CATEGORICAL,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL_CMAP,
    header,
    save,
    use_style,
)
from memlab.render import cell_extent, cell_on_screen, filmstrip_steps, rollout, visibility
from memlab.world import default_layout

TASK = "find_object"
SEED = 0
N_FRAMES = 6


def main() -> None:
    use_style()
    episode = rollout(TASK, SEED)
    layout = default_layout()
    steps = filmstrip_steps(episode, N_FRAMES)

    print(f"  {TASK} seed={SEED}: {len(episode)} steps, success={episode.success}")
    for step, text in episode.event_steps():
        print(f"    t={step:>3}  {text}")

    # Subfigures rather than one gridspec: the filmstrip must not inherit the
    # column split of the row above it, and the map's square aspect otherwise
    # drags whitespace through the whole layout.
    fig = plt.figure(figsize=(15, 8.4))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.5, 1.0])
    ax_map, ax_time = top.subplots(1, 2, width_ratios=[1.0, 1.3])
    film = bottom.subplots(1, N_FRAMES)

    _draw_route(ax_map, episode, layout)
    rows = _memory_rows(episode, layout)
    _draw_timeline(ax_time, episode, rows)
    _draw_filmstrip(film, episode, steps)

    # Summed over rows this would exceed the episode length, so report the tail
    # instead: the stretch after which no searched cabinet is ever on screen.
    searched = [mask for label, mask, mark in rows if label.startswith("cabinet")]
    seen = np.flatnonzero(np.any(searched, axis=0)) if searched else np.array([])
    tail = len(episode) - 1 - int(seen[-1]) if len(seen) else len(episode)
    header(
        fig,
        f"One {TASK.replace('_', ' ')} episode: {len(episode)} steps, "
        f"{len(episode.events)} events",
        f"The bottom row is everything the policy receives — the kitchen above it is "
        f"never shown. From step {int(seen[-1]) if len(seen) else 0} on, no searched "
        f"cabinet is on screen again: for the last {tail} steps "
        f"“already looked there” exists only in memory.",
    )
    save(fig, "09_oracle_rollout")


def _draw_route(ax, episode, layout) -> None:
    ax.imshow(episode.global_frames[-1], extent=cell_extent(), interpolation="nearest")
    path = episode.positions
    segments = np.stack([path[:-1], path[1:]], axis=1)
    ax.add_collection(
        LineCollection(segments, cmap=SEQUENTIAL_CMAP, linewidths=2.4,
                       array=np.linspace(0, 1, len(segments)))
    )
    ax.plot(*path[0], marker="o", ms=8, mfc="none", mec=CATEGORICAL[1], mew=2)
    ax.annotate("start", xy=path[0], xytext=(9, -3), textcoords="offset points",
                fontsize=9, color=CATEGORICAL[1])
    for cid, (r, c) in enumerate(layout.cabinets):
        # Inside the cabinet cell, not below it: the agent's route runs along
        # the floor row directly underneath.
        ax.annotate(str(cid), xy=(c + 0.5, r + 0.5), ha="center", va="center",
                    fontsize=8.5, color="white")
    for step, text in episode.event_steps():
        if not text.startswith("saw"):
            ax.plot(*path[step], marker="o", ms=5, color=CATEGORICAL[1])
    ax.set_xlim(0, 12)
    ax.set_ylim(12, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("a · Where it went (colour = time)")
    ax.set_xlabel("numbers label cabinets; dots mark interactions",
                  fontsize=8.5, color=INK_MUTED)


def _draw_timeline(ax, episode, rows) -> None:
    for i, (label, mask, mark) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.broken_barh(_spans(mask), (y - 0.22, 0.44),
                       facecolors=CATEGORICAL[0], edgecolors="none")
        ax.axhline(y, color=BASELINE, lw=0.8, zorder=0)
        if mark is not None:
            ax.plot(mark, y + 0.30, marker="v", ms=7, color=CATEGORICAL[1], clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in rows][::-1], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, len(episode))
    ax.set_xlabel("step        ▼ the step the fact was established")
    ax.set_title("c · On screen (blue) vs only in memory (blank)")
    ax.grid(False)


def _draw_filmstrip(axes, episode, steps) -> None:
    for ax, step in zip(axes, steps):
        ax.imshow(episode.frames[step], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BASELINE)
        ax.set_title(f"t={step}", fontsize=9.5, pad=4)
        ax.set_xlabel(_caption(episode, step), fontsize=8, color=INK_SECONDARY,
                      labelpad=4)


def _memory_rows(episode, layout) -> list[tuple[str, np.ndarray, int | None]]:
    """One row per fact a policy would have to hold.

    Each mask is gated on the step the fact was established -- a cabinet being
    on screen before the agent searched it is not evidence of anything.
    """
    clock = np.arange(len(episode))
    kind = episode.instruction.replace("bring me the ", "")
    rows: list[tuple[str, np.ndarray, int | None]] = [
        (f"{kind} on screen", visibility(episode, 0), None)
    ]
    for step, text in episode.event_steps():
        if not text.startswith("opened cabinet"):
            continue
        cid = int(text.rsplit(" ", 1)[1])
        on_screen = cell_on_screen(episode.positions, layout.cabinets[cid])
        rows.append((f"cabinet {cid} searched", on_screen & (clock >= step), step))
    return rows


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, width), for broken_barh."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def _caption(episode, step: int) -> str:
    for at, text in episode.event_steps():
        if at == step:
            return text
    return episode.subtasks[step]


if __name__ == "__main__":
    main()
