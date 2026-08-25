"""A scripted controller with full state access, used to generate demos.

The oracle is deliberately *not* optimal. It searches cabinets in a random order
rather than walking straight to the right one, because the whole point of the
dataset is to contain episodes where the correct next action is only explicable
from history. A demonstrator that never searches teaches a policy that never
needs to remember.

Motion is a smoothed proportional controller rather than bang-bang, for two
reasons: it keeps the agent from juddering into counters, and it makes the
action stream band-limited, which is the property FAST's DCT stage exploits.
See `fastlab.data.synthetic_chunks` for the same argument on synthetic data.

Frames are 64x64x3 uint8, so a 500-step episode is ~6 MB. Generating a large
dataset means writing to disk or setting `keep_frames=False` and re-rendering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tasks import Primitive, TaskSpec
from .world import GRID, MAX_SPEED, Event, Kitchen, Layout, standing_cell

SMOOTHING = 0.6        # velocity low-pass; higher is smoother and slower
GAIN = 1.6             # proportional gain on position error
WAYPOINT_TOL = 0.18
ARRIVE_TOL = 0.09
MAX_RETRIES = 3


@dataclass(frozen=True)
class Episode:
    """One demonstration. `frames[t]` is what the agent saw before `actions[t]`."""

    instruction: str
    task: str
    frames: np.ndarray            # (T, 64, 64, 3) uint8, what the agent saw
    global_frames: np.ndarray     # (T, 96, 96, 3) uint8, the god view, if kept
    actions: np.ndarray           # (T, 3) float64
    positions: np.ndarray         # (T, 2) float64
    subtasks: tuple[str, ...]     # length T, the high-level label in force
    visible: tuple[frozenset[int], ...]   # object ids on screen at each step
    events: tuple[Event, ...]
    success: bool

    def __len__(self) -> int:
        return len(self.actions)

    def event_steps(self) -> tuple[tuple[int, str], ...]:
        """(step, text) for each event, clipped to the recorded range."""
        return tuple((min(e.step, len(self) - 1), e.text) for e in self.events)


class _Recorder:
    """Drives the kitchen and accumulates the arrays an Episode needs."""

    def __init__(
        self, kitchen: Kitchen, keep_frames: bool, cap: int, keep_global: bool = False
    ) -> None:
        self.k = kitchen
        self.keep_frames = keep_frames
        self.keep_global = keep_global
        self.cap = cap
        self.frames: list[np.ndarray] = []
        self.globals: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.subtasks: list[str] = []
        self.visible: list[frozenset[int]] = []
        self.obs = kitchen.observe()
        self.velocity = np.zeros(2)

    @property
    def full(self) -> bool:
        return len(self.actions) >= self.cap

    def act(self, action: np.ndarray, label: str) -> None:
        # Everything recorded here describes the state *before* the action, so
        # frames[t], visible[t] and actions[t] line up as one decision.
        if self.keep_frames:
            self.frames.append(self.obs)
        if self.keep_global:
            self.globals.append(self.k.render_global())
        self.visible.append(self.k.visible_objects())
        self.actions.append(np.asarray(action, dtype=np.float64))
        self.positions.append(self.k.pos.copy())
        self.subtasks.append(label)
        self.obs = self.k.step(action)


def run_episode(
    spec: TaskSpec,
    layout: Layout | None = None,
    keep_frames: bool = True,
    keep_global: bool = False,
) -> Episode:
    """Execute `spec`'s plan and return the resulting demonstration.

    `keep_global` additionally records the whole-kitchen view at every step,
    which the agent never sees. It is for figures and playback only -- training
    on it would be cheating.
    """
    kitchen = Kitchen(layout)
    kitchen.reset(spec.objects)
    rec = _Recorder(kitchen, keep_frames, spec.horizon, keep_global)

    for prim in spec.plan:
        if rec.full:
            break
        _execute(rec, prim)

    return Episode(
        instruction=spec.instruction,
        task=spec.name,
        frames=_stack(rec.frames, (0, 64, 64, 3)),
        global_frames=_stack(rec.globals, (0, 96, 96, 3)),
        actions=np.stack(rec.actions) if rec.actions else np.empty((0, 3)),
        positions=np.stack(rec.positions) if rec.positions else np.empty((0, 2)),
        subtasks=tuple(rec.subtasks),
        visible=tuple(rec.visible),
        events=tuple(kitchen.events),
        success=spec.success(kitchen),
    )


def _stack(frames: list[np.ndarray], empty: tuple[int, ...]) -> np.ndarray:
    return np.stack(frames) if frames else np.empty(empty, dtype=np.uint8)


def _execute(rec: _Recorder, prim: Primitive) -> bool:
    """Approach the primitive's cell, fire the gripper, retry if nothing moved.

    Never pulse without a successful approach. An unapproached pulse still fires
    at whatever happens to be in reach, which is how a "pick up the soap" step
    once ended up reopening the cabinet the agent had just closed.
    """
    for _ in range(MAX_RETRIES):
        if rec.full:
            return False
        if not _approach(rec, prim):
            return False
        before = _local_state(rec.k, prim.cell)
        _pulse(rec, prim)
        if _local_state(rec.k, prim.cell) != before:
            return True
        # Reached but nothing changed -- usually stopped a hair short of reach.
        # Re-approaching is what a real scripted policy does, and it is what
        # puts recoverable failures into the dataset.
    return False


def _local_state(k: Kitchen, cell: tuple[int, int]) -> tuple:
    """What a correct interaction at `cell` would have to change.

    Scoped to the target rather than to the global event log, so an accidental
    interaction somewhere else in the kitchen cannot be mistaken for success.
    """
    cab = next((c for c in k.cabinets.values() if c.cell == cell), None)
    resting = tuple(sorted(i for i, o in k.objects.items() if o.cell == cell))
    stowed = (
        tuple(sorted(i for i, o in k.objects.items() if o.container == cab.cid))
        if cab is not None
        else ()
    )
    return (k.holding, None if cab is None else cab.is_open, resting, stowed)


def _approach(rec: _Recorder, prim: Primitive) -> bool:
    """Walk to a spot from which `prim.cell` is the unambiguous target."""
    stand = standing_cell(rec.k.layout, prim.cell, rec.k.pos)
    if stand is None:
        return False
    path = _bfs(rec.k.layout, _cell_of(rec.k.pos), stand)
    if path is None:
        return False
    for i, cell in enumerate(path):
        target = np.array([cell[1] + 0.5, cell[0] + 0.5])
        tol = ARRIVE_TOL if i == len(path) - 1 else WAYPOINT_TOL
        _drive_to(rec, target, tol, prim.label)
    return True


def _drive_to(rec: _Recorder, target: np.ndarray, tol: float, label: str) -> None:
    stall = 0
    while not rec.full:
        error = target - rec.k.pos
        distance = float(np.linalg.norm(error))
        if distance <= tol:
            return
        desired = np.clip(error * GAIN, -MAX_SPEED, MAX_SPEED)
        rec.velocity = SMOOTHING * rec.velocity + (1.0 - SMOOTHING) * desired
        before = rec.k.pos.copy()
        rec.act(np.array([rec.velocity[0], rec.velocity[1], 0.0]), label)
        # Blocked against geometry: give up on this waypoint rather than grind.
        stall = stall + 1 if np.linalg.norm(rec.k.pos - before) < 1e-3 else 0
        if stall > 5:
            return


def _pulse(rec: _Recorder, prim: Primitive) -> None:
    """Gripper commands are edge-triggered, so a pulse is fire-then-neutral."""
    level = 1.0 if prim.kind == "grasp" else -1.0
    if not rec.full:
        rec.act(np.array([0.0, 0.0, level]), prim.label)
    if not rec.full:
        rec.act(np.array([0.0, 0.0, 0.0]), prim.label)


def _cell_of(pos: np.ndarray) -> tuple[int, int]:
    return int(np.floor(pos[1])), int(np.floor(pos[0]))


def _bfs(
    layout: Layout, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Shortest 4-connected walkable path, returned as cells excluding start."""
    if start == goal:
        return []
    walkable = layout.walkable
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in seen or not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                continue
            if not walkable[nxt]:
                continue
            seen.add(nxt)
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return list(reversed(path[:-1]))
            queue.append(nxt)
    return None


def collect(
    task: str,
    n_episodes: int,
    seed: int = 0,
    keep_frames: bool = True,
    keep_global: bool = False,
    **task_kwargs,
) -> list[Episode]:
    """Generate `n_episodes` demonstrations of `task`."""
    from .tasks import make_task

    rng = np.random.default_rng(seed)
    layout = Kitchen().layout
    return [
        run_episode(make_task(task, rng, **task_kwargs), layout, keep_frames, keep_global)
        for _ in range(n_episodes)
    ]
