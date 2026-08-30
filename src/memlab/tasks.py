"""Task generators, each built so that memory is load-bearing.

A task is only a memory task if the correct action at time t depends on
something no longer visible at time t. Both tasks here are constructed to force
that, in the two ways MEM distinguishes:

- `find_object` needs *short-to-mid horizon* recall of an observation: you
  opened cabinet 3 and glimpsed the bread, and now you must act on that glimpse
  after walking away from it.
- `tidy_up` needs *long horizon* progress tracking: which objects have already
  been stowed, and which doors are still hanging open. Nothing on screen tells
  you; the state lives entirely in what you did earlier.

Each task returns the object placement, an instruction string, an oracle plan,
and a success predicate. The plan is what makes the demonstrations *searchy*:
the oracle deliberately does not teleport to the right cabinet, because a
demonstration with no search in it teaches a policy that needs no memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .world import Kitchen, Obj, default_layout, reachable

_LAYOUT = default_layout()


def _usable(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Keep only cells the agent can actually act on without ambiguity.

    (10, 1) fails this: it sits equidistant from the sink, so the simulator
    would resolve an interaction there to the wrong cell. Filtering here means a
    layout edit that strands a cell shows up as fewer spots, not as episodes
    that quietly go wrong.
    """
    return tuple(c for c in cells if reachable(_LAYOUT, c))


TABLE_CELLS = _usable(((10, 4), (10, 5), (10, 6)))
COUNTER_SPOTS = _usable(((10, 1), (10, 2), (10, 3), (10, 8), (10, 9)))
DISTRACTORS = ("mug", "plate", "kettle", "soap", "butter")


@dataclass(frozen=True)
class Primitive:
    """One oracle motion. `label` is the high-level subtask it serves."""

    kind: str                    # "grasp" | "release"
    cell: tuple[int, int]        # the cell to stand next to and act on
    label: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    objects: dict[int, Obj]
    plan: tuple[Primitive, ...]
    success: Callable[[Kitchen], bool]
    horizon: int


def find_object(
    rng: np.random.Generator, n_cabinets: int = 5, clutter: int = 0
) -> TaskSpec:
    """Search cabinets for one target, then put it on the table.

    `clutter` puts that many identical items in the first cabinet searched.
    That is the paper's own compression example -- several separately-observed
    bowls that should be remembered as "three bowls in the top right cabinet" --
    and it is the only source of redundancy in these tasks, so it is what
    experiment 10 varies to show when compaction can and cannot help.
    """
    target = "bread"
    home = int(rng.integers(0, n_cabinets))

    objects: dict[int, Obj] = {0: Obj(target, container=home)}
    others = list(rng.permutation([c for c in range(n_cabinets) if c != home]))
    for i, cid in enumerate(others[:2]):
        objects[i + 1] = Obj(DISTRACTORS[i], container=int(cid))

    # The oracle searches in a random order and stops when it finds the target,
    # so the number of cabinets opened before the pick varies across episodes.
    order = [int(c) for c in rng.permutation(n_cabinets)]
    order = order[: order.index(home) + 1]

    if clutter and len(order) > 1:
        for j in range(clutter):
            objects[100 + j] = Obj("mug", container=order[0])

    cabinet_cell = _cabinet_cells(n_cabinets)
    steps: list[Primitive] = []
    for cid in order:
        steps.append(Primitive("grasp", cabinet_cell[cid], f"open cabinet {cid}"))
    steps.append(Primitive("grasp", cabinet_cell[home], f"pick up the {target}"))
    drop = TABLE_CELLS[int(rng.integers(0, len(TABLE_CELLS)))]
    steps.append(Primitive("release", drop, f"put the {target} on the table"))

    def success(k: Kitchen) -> bool:
        return k.objects[0].cell in TABLE_CELLS

    return TaskSpec(
        name="find_object",
        instruction=f"bring me the {target}",
        objects=objects,
        plan=tuple(steps),
        success=success,
        horizon=90 + 60 * len(order),
    )


def tidy_up(rng: np.random.Generator, n_items: int = 3, n_cabinets: int = 5) -> TaskSpec:
    """Stow loose items in assigned cabinets and close every door behind you."""
    kinds = [str(k) for k in rng.permutation(list(DISTRACTORS))[:n_items]]
    spots = [COUNTER_SPOTS[i] for i in rng.permutation(len(COUNTER_SPOTS))[:n_items]]
    homes = [int(c) for c in rng.permutation(n_cabinets)[:n_items]]

    objects = {i: Obj(kind, cell=spot) for i, (kind, spot) in enumerate(zip(kinds, spots))}
    cabinet_cell = _cabinet_cells(n_cabinets)

    steps: list[Primitive] = []
    for i, (kind, spot, home) in enumerate(zip(kinds, spots, homes)):
        cab = cabinet_cell[home]
        steps.append(Primitive("grasp", spot, f"pick up the {kind}"))
        steps.append(Primitive("grasp", cab, f"open cabinet {home}"))
        steps.append(Primitive("release", cab, f"put the {kind} in cabinet {home}"))
        steps.append(Primitive("release", cab, f"close cabinet {home}"))

    def success(k: Kitchen) -> bool:
        stowed = all(k.objects[i].container == homes[i] for i in range(n_items))
        shut = all(not c.is_open for c in k.cabinets.values())
        return stowed and shut

    return TaskSpec(
        name="tidy_up",
        instruction="put everything away and close the cabinets",
        objects=objects,
        plan=tuple(steps),
        success=success,
        horizon=140 * n_items,
    )


def _cabinet_cells(n_cabinets: int) -> list[tuple[int, int]]:
    from .world import default_layout

    return list(default_layout().cabinets[:n_cabinets])


TASKS: dict[str, Callable[..., TaskSpec]] = {
    "find_object": find_object,
    "tidy_up": tidy_up,
}


def make_task(name: str, rng: np.random.Generator, **kwargs) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {sorted(TASKS)}")
    return TASKS[name](rng, **kwargs)
