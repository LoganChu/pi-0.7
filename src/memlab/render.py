"""Watching the oracle run, rather than only measuring what it produced.

Two ways to look at an episode:

- `save_gif` writes an animation with the god view and the agent's actual 64x64
  observation side by side, captioned with the subtask in force. Seeing both at
  once is the point -- the left panel is what a reader needs to follow the task,
  the right panel is all the policy will ever get.
- `visibility` turns an episode into the timeline experiment 09 plots: for each
  step, whether the object the agent must act on is on screen at all.

Run it directly to produce a GIF:

    python -m memlab.render --task find_object --seed 0
    python -m memlab.render --task tidy_up --seed 3 --stride 3 --out run.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .oracle import Episode, run_episode
from .tasks import make_task
from .world import CELL_PX, GRID, OBJECT_COLORS

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
CAPTION_PX = 34
GAP_PX = 12
BACKDROP = (252, 252, 251)
INK = (11, 11, 11)
MUTED = (137, 135, 129)


def rollout(task: str, seed: int = 0, **task_kwargs) -> Episode:
    """One episode with everything recorded, ready to visualize."""
    rng = np.random.default_rng(seed)
    spec = make_task(task, rng, **task_kwargs)
    return run_episode(spec, keep_frames=True, keep_global=True)


def visibility(episode: Episode, object_id: int) -> np.ndarray:
    """Boolean per step: was this object drawn in the agent's own view?"""
    return np.array([object_id in v for v in episode.visible], dtype=bool)


def target_id(episode: Episode) -> int:
    """The object the task is about. Object 0 by construction in both tasks."""
    return 0


def cell_on_screen(positions: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    """Boolean per step: is this cell inside the agent's egocentric crop?

    Pure geometry, so it needs only the recorded positions. Used to ask the
    question a memory task turns on -- once the agent walks away, is there
    anything on screen still recording what it did here?
    """
    from .world import FRAME_PX

    half = FRAME_PX // 2
    cx = np.round(positions[:, 0] * CELL_PX)
    cy = np.round(positions[:, 1] * CELL_PX)
    bx, by = cell[1] * CELL_PX + 4, cell[0] * CELL_PX + 4
    return (np.abs(bx - cx) < half) & (np.abs(by - cy) < half)


def filmstrip_steps(episode: Episode, n: int = 6) -> list[int]:
    """Steps worth showing: the start, then the moments something happened."""
    candidates = [0]
    for step, _ in episode.event_steps():
        if step not in candidates:
            candidates.append(step)
    if len(candidates) > n:
        keep = np.linspace(0, len(candidates) - 1, n).round().astype(int)
        candidates = [candidates[i] for i in sorted(set(keep))]
    while len(candidates) < n:
        gaps = np.diff([0, *candidates, len(episode) - 1])
        widest = int(np.argmax(gaps))
        filler = ([0, *candidates, len(episode) - 1][widest] + gaps[widest] // 2)
        if filler in candidates:
            break
        candidates.append(int(filler))
    return sorted(candidates)


def compose_frame(episode: Episode, t: int, scale: int = 4) -> Image.Image:
    """God view and egocentric view side by side, with a caption bar."""
    world = _upscale(episode.global_frames[t], scale)
    ego = _upscale(episode.frames[t], scale)

    width = world.width + GAP_PX + ego.width
    height = max(world.height, ego.height) + CAPTION_PX
    canvas = Image.new("RGB", (width, height), BACKDROP)
    canvas.paste(world, (0, CAPTION_PX))
    # The egocentric frame is smaller than the kitchen; centring it vertically
    # stops the panel from looking clipped at the bottom.
    canvas.paste(ego, (world.width + GAP_PX, CAPTION_PX + (world.height - ego.height) // 2))

    draw = ImageDraw.Draw(canvas)
    draw.text((2, 4), f"t={t:>3}  {episode.subtasks[t]}", fill=INK)
    draw.text((2, 19), "kitchen (never shown to the agent)", fill=MUTED)
    draw.text((world.width + GAP_PX + 2, 19), "what the agent sees", fill=MUTED)
    return canvas


def save_gif(
    episode: Episode,
    path: str | Path | None = None,
    stride: int = 2,
    fps: int = 20,
    scale: int = 4,
) -> Path:
    """Write the episode as an animated GIF and report the path."""
    if not len(episode.global_frames):
        raise ValueError("episode has no global frames; run with keep_global=True")

    steps = range(0, len(episode), stride)
    images = [compose_frame(episode, t, scale) for t in steps]
    out = Path(path) if path else FIGURES_DIR / f"09_{episode.task}_rollout.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"  wrote {out}  ({len(images)} frames, {out.stat().st_size / 1e6:.1f} MB)")
    return out


def _upscale(frame: np.ndarray, scale: int) -> Image.Image:
    # Nearest neighbour on purpose: these are 8-px cells, and smoothing them
    # would invent detail the agent's observation does not contain.
    img = Image.fromarray(frame)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def object_color(kind: str) -> tuple[float, float, float]:
    """The world's colour for an object kind, normalized for matplotlib."""
    return tuple(c / 255 for c in OBJECT_COLORS[kind])


def cell_extent() -> tuple[float, float, float, float]:
    """imshow extent that puts a global frame into cell coordinates.

    With this, agent positions can be plotted straight onto the rendered
    kitchen without converting to pixels.
    """
    return (0.0, float(GRID), float(GRID), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an oracle episode as a GIF.")
    parser.add_argument("--task", default="find_object", choices=["find_object", "tidy_up"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2, help="keep every Nth step")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    episode = rollout(args.task, args.seed)
    print(f"  {args.task} seed={args.seed}: {len(episode)} steps, "
          f"success={episode.success}")
    for step, text in episode.event_steps():
        print(f"    t={step:>3}  {text}")
    save_gif(episode, args.out, stride=args.stride, fps=args.fps, scale=args.scale)


if __name__ == "__main__":
    main()
