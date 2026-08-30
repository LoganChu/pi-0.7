"""A 2-D kitchen that cannot be solved without memory.

The point of this world is *partial observability*. The agent sees an egocentric
64x64 crop, cabinets hide their contents until opened, and the objects a task
needs are placed out of sight. So the only way to know where the bread is, is to
have opened that cabinet earlier and remembered it -- which is exactly the
pressure MEM's memory is meant to relieve.

Geometry is continuous even though the layout is a grid: the agent holds a float
(x, y) and moves by a velocity command, so action chunks are smooth trajectories
that FAST can tokenize, not discrete grid hops.

    cell (r, c) covers x in [c, c+1) and y in [r, r+1)
    action is (dx, dy, grip), all continuous
        grip > +0.5  -> grasp / open   (fires on rising edge)
        grip < -0.5  -> release / close
        otherwise    -> no-op

The world object is mutable, as simulators are. Everything it hands back --
frames, event tuples, layouts -- is immutable or a fresh array.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

CELL_PX = 8
VIEW_CELLS = 8
FRAME_PX = CELL_PX * VIEW_CELLS  # 64
GRID = 12

MAX_SPEED = 0.22          # cells per step
AGENT_RADIUS = 0.28
REACH = 1.45              # centre-to-centre distance the agent can interact over
GRIP_HI, GRIP_LO = 0.5, -0.5


class Cell(IntEnum):
    FLOOR = 0
    WALL = 1
    COUNTER = 2
    STOVE = 3
    SINK = 4
    TABLE = 5
    CABINET = 6


PALETTE: dict[int, tuple[int, int, int]] = {
    Cell.FLOOR: (246, 245, 241),
    Cell.WALL: (120, 118, 110),
    Cell.COUNTER: (206, 203, 193),
    Cell.STOVE: (70, 68, 66),
    Cell.SINK: (150, 175, 195),
    Cell.TABLE: (196, 168, 120),
}
CABINET_CLOSED = (150, 120, 86)
CABINET_OPEN = (96, 74, 52)
AGENT_COLOR = (42, 120, 214)
OUT_OF_BOUNDS = (86, 85, 80)

# Object colors are pulled apart deliberately -- a 4x4 px blob has to be
# identifiable from a 64x64 frame, so neighbouring kinds never share a hue.
OBJECT_COLORS: dict[str, tuple[int, int, int]] = {
    "bread": (235, 140, 60),
    "cheese": (237, 190, 40),
    "butter": (250, 240, 190),
    "pan": (45, 45, 48),
    "plate": (250, 250, 252),
    "mug": (232, 123, 164),
    "soap": (27, 175, 122),
    "kettle": (74, 58, 167),
}


@dataclass(frozen=True)
class Event:
    """One thing that happened, in a form the text-memory stage can consume.

    Carries the structured fields as well as the sentence, so the memory writer
    reads `event.cabinet` rather than parsing `"opened cabinet 3"` back apart.
    """

    step: int
    kind: str
    text: str
    obj: str | None = None
    cabinet: int | None = None
    surface: str | None = None


@dataclass
class Obj:
    kind: str
    container: int | None = None          # cabinet id, if stowed
    cell: tuple[int, int] | None = None   # resting cell, if out in the open
    held: bool = False


@dataclass
class Cabinet:
    cid: int
    cell: tuple[int, int]
    is_open: bool = False


@dataclass(frozen=True)
class Layout:
    """Static kitchen geometry. Built once, shared by every episode."""

    grid: np.ndarray                        # (GRID, GRID) int8 of Cell
    cabinets: tuple[tuple[int, int], ...]   # cell of each cabinet, by id
    spawn: tuple[float, float]

    @property
    def walkable(self) -> np.ndarray:
        return self.grid == Cell.FLOOR


def default_layout() -> Layout:
    """A kitchen with five cabinets, a stove, a sink, a table and an island.

    The island exists so navigation is not trivially straight-line -- the oracle
    has to path around it, which puts curvature into the action chunks.
    """
    g = np.full((GRID, GRID), Cell.FLOOR, dtype=np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = Cell.WALL

    # Top run: cabinets at odd columns, counter between them.
    cabinets: list[tuple[int, int]] = []
    for c in range(1, 11):
        if c % 2 == 1:
            g[1, c] = Cell.CABINET
            cabinets.append((1, c))
        else:
            g[1, c] = Cell.COUNTER

    g[4, 1] = g[5, 1] = Cell.STOVE
    g[7, 1] = g[8, 1] = Cell.SINK
    for c in (1, 2, 3, 7, 8, 9, 10):
        g[10, c] = Cell.COUNTER
    g[10, 4] = g[10, 5] = g[10, 6] = Cell.TABLE
    g[5, 5] = g[5, 6] = g[6, 5] = g[6, 6] = Cell.COUNTER  # island

    return Layout(grid=g, cabinets=tuple(cabinets), spawn=(3.5, 3.5))


class Kitchen:
    """Mutable simulator. `reset` returns the first frame, `step` the next."""

    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout or default_layout()
        self._static = _render_static(self.layout)
        self.objects: dict[int, Obj] = {}
        self.cabinets: dict[int, Cabinet] = {}
        self.pos = np.array(self.layout.spawn, dtype=np.float64)
        self.holding: int | None = None
        self.events: list[Event] = []
        self.t = 0
        self._prev_grip = 0.0

    # -- setup ------------------------------------------------------------

    def reset(
        self,
        objects: dict[int, Obj],
        spawn: tuple[float, float] | None = None,
    ) -> np.ndarray:
        self.objects = {k: replace(v) for k, v in objects.items()}
        self.cabinets = {i: Cabinet(i, cell) for i, cell in enumerate(self.layout.cabinets)}
        self.pos = np.array(spawn or self.layout.spawn, dtype=np.float64)
        self.holding = None
        self.events = []
        self.t = 0
        self._prev_grip = 0.0
        return self.observe()

    # -- dynamics ---------------------------------------------------------

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply one (dx, dy, grip) command and return the next frame."""
        a = np.asarray(action, dtype=np.float64).reshape(3)
        self._move(a[:2])
        self._maybe_interact(float(a[2]))
        self._prev_grip = float(a[2])
        self.t += 1
        return self.observe()

    def _move(self, velocity: np.ndarray) -> None:
        v = np.clip(velocity, -MAX_SPEED, MAX_SPEED)
        # Axis-separated resolution, so sliding along a counter feels natural
        # rather than sticking the moment one axis is blocked.
        for axis in (0, 1):
            candidate = self.pos.copy()
            candidate[axis] += v[axis]
            if self._free(candidate):
                self.pos = candidate

    def _free(self, p: np.ndarray) -> bool:
        for dx in (-AGENT_RADIUS, AGENT_RADIUS):
            for dy in (-AGENT_RADIUS, AGENT_RADIUS):
                c, r = int(np.floor(p[0] + dx)), int(np.floor(p[1] + dy))
                if not (0 <= r < GRID and 0 <= c < GRID):
                    return False
                if self.layout.grid[r, c] != Cell.FLOOR:
                    return False
        return True

    def _maybe_interact(self, grip: float) -> None:
        # Edge-triggered: holding the gripper open must not re-fire every step.
        was_neutral = GRIP_LO <= self._prev_grip <= GRIP_HI
        if not was_neutral:
            return
        if grip > GRIP_HI:
            self._grasp()
        elif grip < GRIP_LO:
            self._release()

    def _target_cell(self) -> tuple[int, int] | None:
        """Nearest interactable cell within reach of the agent."""
        return nearest_interactable(self.layout, self.pos)

    def _cabinet_at(self, cell: tuple[int, int]) -> Cabinet | None:
        for cab in self.cabinets.values():
            if cab.cell == cell:
                return cab
        return None

    def _grasp(self) -> None:
        cell = self._target_cell()
        if cell is None:
            return
        cab = self._cabinet_at(cell)

        if cab is not None and not cab.is_open:
            cab.is_open = True
            self._log("open", f"opened cabinet {cab.cid}", cabinet=cab.cid)
            seen = [o.kind for o in self.objects.values() if o.container == cab.cid]
            for kind in seen:
                self._log("see", f"saw {kind} in cabinet {cab.cid}", obj=kind, cabinet=cab.cid)
            if not seen:
                self._log("see", f"cabinet {cab.cid} was empty", cabinet=cab.cid)
            return

        if self.holding is not None:
            return
        oid = self._pick_target(cell, cab)
        if oid is None:
            return
        obj = self.objects[oid]
        obj.held, obj.container, obj.cell = True, None, None
        self.holding = oid
        self._log("pick", f"picked up {obj.kind}", obj=obj.kind)

    def _pick_target(self, cell: tuple[int, int], cab: Cabinet | None) -> int | None:
        if cab is not None:
            if not cab.is_open:
                return None
            for oid, o in self.objects.items():
                if o.container == cab.cid:
                    return oid
            return None
        for oid, o in self.objects.items():
            if o.cell == cell:
                return oid
        return None

    def _release(self) -> None:
        cell = self._target_cell()
        if cell is None:
            return
        cab = self._cabinet_at(cell)

        if self.holding is None:
            if cab is not None and cab.is_open:
                cab.is_open = False
                self._log("close", f"closed cabinet {cab.cid}", cabinet=cab.cid)
            return

        obj = self.objects[self.holding]
        if cab is not None:
            if not cab.is_open:
                return
            obj.container, obj.cell = cab.cid, None
            self._log("place", f"put {obj.kind} in cabinet {cab.cid}", obj=obj.kind, cabinet=cab.cid)
        else:
            surface = Cell(int(self.layout.grid[cell])).name.lower()
            obj.container, obj.cell = None, cell
            self._log("place", f"put {obj.kind} on the {surface}", obj=obj.kind, surface=surface)
        obj.held = False
        self.holding = None

    def _log(self, kind: str, text: str, **fields) -> None:
        self.events.append(Event(step=self.t, kind=kind, text=text, **fields))

    # -- observation ------------------------------------------------------

    def observe(self) -> np.ndarray:
        """Egocentric (64, 64, 3) uint8 crop, centred on the agent."""
        world = self.render_global()
        half = FRAME_PX // 2
        cx = int(round(float(self.pos[0]) * CELL_PX))
        cy = int(round(float(self.pos[1]) * CELL_PX))

        frame = np.empty((FRAME_PX, FRAME_PX, 3), dtype=np.uint8)
        frame[:] = OUT_OF_BOUNDS
        y0, x0 = cy - half, cx - half
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(world.shape[0], y0 + FRAME_PX), min(world.shape[1], x0 + FRAME_PX)
        if sy1 > sy0 and sx1 > sx0:
            frame[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = world[sy0:sy1, sx0:sx1]

        # The agent is always dead centre; drawing it after the crop keeps it
        # from being clipped at the world edge.
        frame[half - 3 : half + 3, half - 3 : half + 3] = AGENT_COLOR
        if self.holding is not None:
            frame[half - 1 : half + 1, half - 1 : half + 1] = OBJECT_COLORS[
                self.objects[self.holding].kind
            ]
        return frame

    def visible_objects(self) -> frozenset[int]:
        """Object ids whose marker is actually drawn in the current frame.

        This is the honest test of whether a task needs memory: if the thing the
        agent must act on is not in this set at the moment it must act, the
        information has to come from somewhere other than the observation.

        A held object counts as visible -- it is drawn on the agent.
        """
        half = FRAME_PX // 2
        cx = int(round(float(self.pos[0]) * CELL_PX))
        cy = int(round(float(self.pos[1]) * CELL_PX))
        seen = set()
        for oid, obj in self.objects.items():
            if obj.held:
                seen.add(oid)
                continue
            cell = self._visible_cell(obj)
            if cell is None:
                continue
            bx, by = cell[1] * CELL_PX + 4, cell[0] * CELL_PX + 4  # blob centre
            if abs(bx - cx) < half and abs(by - cy) < half:
                seen.add(oid)
        return frozenset(seen)

    def render_global(self) -> np.ndarray:
        """Full-kitchen view. Only for figures -- the agent never sees this."""
        img = self._static.copy()
        for cab in self.cabinets.values():
            r, c = cab.cell
            _fill(img, r, c, CABINET_OPEN if cab.is_open else CABINET_CLOSED)
        for obj in self.objects.values():
            cell = self._visible_cell(obj)
            if cell is not None:
                _blob(img, cell[0], cell[1], OBJECT_COLORS[obj.kind])
        return img

    def _visible_cell(self, obj: Obj) -> tuple[int, int] | None:
        """Where an object shows up, or None if it is hidden or carried."""
        if obj.held:
            return None
        if obj.cell is not None:
            return obj.cell
        if obj.container is not None and self.cabinets[obj.container].is_open:
            return self.cabinets[obj.container].cell
        return None


# -- reach geometry -------------------------------------------------------
#
# Both the simulator and anything that plans against it have to agree on which
# cell an interaction lands on, so that rule lives here once rather than being
# re-derived (and mis-derived) by callers.


def nearest_interactable(layout: Layout, pos: np.ndarray) -> tuple[int, int] | None:
    """The cell a `grasp`/`release` issued from `pos` would act on."""
    best, best_d = None, REACH
    x, y = float(pos[0]), float(pos[1])
    for r in range(GRID):
        for c in range(GRID):
            if layout.grid[r, c] in (Cell.FLOOR, Cell.WALL):
                continue
            d = float(np.hypot((c + 0.5) - x, (r + 0.5) - y))
            if d < best_d:
                best, best_d = (r, c), d
    return best


def standing_cell(
    layout: Layout,
    target: tuple[int, int],
    from_pos: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """A walkable cell from which `target` is unambiguously the nearest one.

    Being *in reach* of the target is not enough. The simulator acts on whatever
    is closest, so a spot that is equidistant from the target and something else
    -- the counter at (10, 1) and the sink at (8, 1) are exactly 1.0 from the
    floor cell between them -- would silently act on the wrong cell. Requiring a
    strict margin makes that case return None instead, so callers can see it.
    """
    tc = np.array([target[1] + 0.5, target[0] + 0.5])
    best, best_d = None, np.inf
    for r in range(GRID):
        for c in range(GRID):
            if layout.grid[r, c] != Cell.FLOOR:
                continue
            centre = np.array([c + 0.5, r + 0.5])
            if float(np.linalg.norm(centre - tc)) > REACH:
                continue
            if not _unambiguous(layout, centre, target):
                continue
            ref = tc if from_pos is None else np.asarray(from_pos, dtype=np.float64)
            d = float(np.linalg.norm(centre - ref))
            if d < best_d:
                best, best_d = (r, c), d
    return best


def _unambiguous(layout: Layout, pos: np.ndarray, target: tuple[int, int]) -> bool:
    """True if `target` is nearest to `pos` by a margin no float noise can flip."""
    best_d, runner_up = np.inf, np.inf
    x, y = float(pos[0]), float(pos[1])
    for r in range(GRID):
        for c in range(GRID):
            if layout.grid[r, c] in (Cell.FLOOR, Cell.WALL):
                continue
            d = float(np.hypot((c + 0.5) - x, (r + 0.5) - y))
            if (r, c) == target:
                best_d = d
            elif d < runner_up:
                runner_up = d
    return best_d <= REACH and runner_up > best_d + 1e-6


def reachable(layout: Layout, target: tuple[int, int]) -> bool:
    """Whether any spot exists from which `target` can be acted on cleanly."""
    return standing_cell(layout, target) is not None


def _fill(img: np.ndarray, r: int, c: int, color: tuple[int, int, int]) -> None:
    img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX] = color


def _blob(img: np.ndarray, r: int, c: int, color: tuple[int, int, int]) -> None:
    y, x = r * CELL_PX + 2, c * CELL_PX + 2
    img[y : y + 4, x : x + 4] = color


def _render_static(layout: Layout) -> np.ndarray:
    img = np.zeros((GRID * CELL_PX, GRID * CELL_PX, 3), dtype=np.uint8)
    for r in range(GRID):
        for c in range(GRID):
            kind = int(layout.grid[r, c])
            _fill(img, r, c, PALETTE.get(kind, CABINET_CLOSED))
    return img
