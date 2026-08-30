"""MEM's long-horizon memory: what happened, written down as compressed text.

The two halves of MEM store history in deliberately different ways. The video
encoder (`encoder.py`) keeps fine detail about the last few seconds. This keeps
the rest -- minutes of it -- as natural language, because language is what stays
small when you summarize it.

The paper's mechanism (sec. III-B): the high-level policy emits a memory update
alongside each subtask, and the training labels for those updates come from an
off-the-shelf LLM handed the subtask annotations plus a success/failure flag,
instructed to "remove or compress information in the language memory whenever
appropriate" -- their example being that three individually-described bowls
should be remembered as "three bowls in the top right cabinet".

WHAT IS AND IS NOT THE POLICY'S MEMORY
--------------------------------------
`Belief` here is a structured state used to *generate labels*, with access to
ground truth from the simulator -- exactly the position the paper's LLM is in.
The policy's memory is the rendered **text**, and nothing else. Queries used for
evaluation run against the belief only because rendering is faithful to it: a
fact dropped by compaction is dropped from both. `llm_memory.py` swaps the
template renderer for a real pretrained LM, which is the paper's actual setup.

Compaction is where the interesting behaviour lives, so it is separable:
`MemoryConfig(compact=False)` degrades to appending every event verbatim, which
is the baseline experiment 10 measures against.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

from .world import Event

NUMBER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine",
}
HELD = "held"


def word_count(text: str) -> int:
    """Default token counter. `memory_tokenizer.py` supplies a real one."""
    return len(text.split())


@dataclass(frozen=True)
class MemoryConfig:
    """How the memory is written, and how big it is allowed to get."""

    compact: bool = True
    max_tokens: int = 48
    protected: tuple[str, ...] = ()   # object kinds the instruction mentions
    count_tokens: Callable[[str], int] = word_count


@dataclass
class Belief:
    """Everything the memory currently claims to know.

    Mutable while being written, then frozen into text. `searched` maps a
    cabinet to the object kinds seen inside it, so an empty list means "opened
    it, nothing there" -- which is a fact worth keeping, not an absence of one.

    `placements` is kind -> place -> how many, rather than kind -> place. The
    multiplicity is the whole point of the paper's compression example: three
    bowls have to be counted before they can be summarized as "three bowls".
    """

    searched: dict[int, list[str]] = field(default_factory=dict)
    placements: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped: int = 0                  # facts compaction had to discard

    def copy(self) -> "Belief":
        return Belief(
            searched={k: list(v) for k, v in self.searched.items()},
            placements={k: dict(v) for k, v in self.placements.items()},
            dropped=self.dropped,
        )

    # -- writing ---------------------------------------------------------

    def add(self, kind: str, place: str, n: int = 1) -> None:
        places = self.placements.setdefault(kind, {})
        places[place] = places.get(place, 0) + n

    def remove_one(self, kind: str, place: str | None = None) -> None:
        places = self.placements.get(kind)
        if not places:
            return
        src = place if place in places else max(places, key=lambda p: places[p])
        places[src] -= 1
        if places[src] <= 0:
            del places[src]
        if not places:
            del self.placements[kind]

    def forget_cabinet(self, cabinet: int) -> None:
        """Drop what was believed about one cabinet's contents.

        Called when it is reopened: a fresh look supersedes the old one, and
        without this a cabinet opened twice would count its contents twice.
        """
        place = f"cabinet {cabinet}"
        for kind in list(self.placements):
            if place in self.placements[kind]:
                del self.placements[kind][place]
                if not self.placements[kind]:
                    del self.placements[kind]

    # -- queries the evaluation asks ------------------------------------

    def knows_searched(self, cabinet: int) -> bool:
        return cabinet in self.searched

    def location_of(self, kind: str) -> str | None:
        places = self.placements.get(kind)
        if not places:
            return None
        if HELD in places:
            return HELD
        return max(places, key=lambda p: places[p])

    @property
    def n_facts(self) -> int:
        return len(self.searched) + len(self.placements)


class TextMemory:
    """The running memory: absorb events, render compressed text on demand."""

    def __init__(self, cfg: MemoryConfig | None = None) -> None:
        self.cfg = cfg or MemoryConfig()
        self.belief = Belief()
        self.verbatim: list[str] = []

    def absorb(self, event: Event) -> None:
        """Fold one observation into the belief."""
        self.verbatim.append(event.text)
        belief = self.belief

        if event.kind == "open" and event.cabinet is not None:
            belief.searched[event.cabinet] = []
            belief.forget_cabinet(event.cabinet)
        elif event.kind == "see" and event.cabinet is not None:
            if event.obj is not None:
                belief.searched.setdefault(event.cabinet, []).append(event.obj)
                belief.add(event.obj, f"cabinet {event.cabinet}")
        elif event.kind == "pick" and event.obj is not None:
            # The object is no longer where it was, so that fact is now stale
            # rather than merely old -- the paper's "remove ... whenever
            # appropriate" applies to contradiction, not just to length.
            for contents in belief.searched.values():
                if event.obj in contents:
                    contents.remove(event.obj)
            belief.remove_one(event.obj)
            belief.add(event.obj, HELD)
        elif event.kind == "place" and event.obj is not None:
            belief.remove_one(event.obj, HELD)
            if event.cabinet is not None:
                belief.searched.setdefault(event.cabinet, []).append(event.obj)
                belief.add(event.obj, f"cabinet {event.cabinet}")
            else:
                belief.add(event.obj, f"the {event.surface}")

    def render(self) -> str:
        """The memory as the policy would hold it: one line of text."""
        if not self.cfg.compact:
            return " ".join(f"{s}." for s in self.verbatim)
        working = self.belief.copy()
        text = _render(working)
        if self.cfg.count_tokens(text) > self.cfg.max_tokens:
            text = _compact_to_budget(working, self.cfg)
            self.belief.dropped = working.dropped
        return text

    def snapshot(self) -> Belief:
        """The belief as rendering left it, for evaluation queries."""
        working = self.belief.copy()
        if self.cfg.compact and self.cfg.count_tokens(_render(working)) > self.cfg.max_tokens:
            _compact_to_budget(working, self.cfg)
        return working


# -- rendering -----------------------------------------------------------


def _render(belief: Belief) -> str:
    clauses = [c for c in (_searched_clause(belief), *_location_clauses(belief)) if c]
    return " ".join(f"{c}." for c in clauses) if clauses else "nothing yet."


def _searched_clause(belief: Belief) -> str:
    """"searched cabinets 0, 2 and 4, 0 and 4 empty" -- one clause, not N."""
    if not belief.searched:
        return ""
    ids = sorted(belief.searched)
    empty = [str(c) for c in ids if not belief.searched[c]]
    clause = f"searched {_plural('cabinet', len(ids))} {_join(str(c) for c in ids)}"
    if empty:
        clause += f", {_join(empty)} empty" if len(empty) < len(ids) else ", all empty"
    return clause


def _location_clauses(belief: Belief) -> Iterable[str]:
    """One clause per place, with same-kind items counted rather than listed.

    This is the paper's own example: three separately-observed bowls become
    "three bowls in cabinet 2" rather than three sentences.
    """
    by_place: dict[str, dict[str, int]] = {}
    for kind, places in belief.placements.items():
        for place, n in places.items():
            by_place.setdefault(place, {})[kind] = by_place.get(place, {}).get(kind, 0) + n

    for place, counts in by_place.items():
        parts = [
            kind if n == 1 else f"{NUMBER_WORDS.get(n, str(n))} {kind}s"
            for kind, n in sorted(counts.items())
        ]
        if place == HELD:
            yield f"holding the {_join(parts)}"
        else:
            yield f"{_join(parts)} in {place}" if place.startswith("cabinet") \
                else f"{_join(parts)} on {place}"


def _compact_to_budget(belief: Belief, cfg: MemoryConfig) -> str:
    """Drop facts until the text fits, least useful first.

    Order is deliberate and is the whole policy question in miniature: an
    object the instruction never mentions is worth less than knowing which
    cabinets have already been searched, because the latter is what stops the
    agent repeating work.
    """
    droppable = [
        kind for kind in belief.placements
        if kind not in cfg.protected and belief.location_of(kind) != HELD
    ]
    for kind in droppable:
        text = _render(belief)
        if cfg.count_tokens(text) <= cfg.max_tokens:
            return text
        del belief.placements[kind]
        belief.dropped += 1

    # Still over budget: give up the oldest searched cabinets, which is a real
    # loss of capability -- the agent may re-search them.
    while len(belief.searched) > 1 and cfg.count_tokens(_render(belief)) > cfg.max_tokens:
        del belief.searched[min(belief.searched)]
        belief.dropped += 1
    return _render(belief)


def _join(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _plural(word: str, n: int) -> str:
    return word if n == 1 else f"{word}s"


# -- per-episode trace ---------------------------------------------------


@dataclass(frozen=True)
class Subtask:
    """One unit of the episode as a memory writer sees it.

    This is the input the paper's labeller gets -- subtask annotations plus a
    success flag, not pixels. A subtask that logged no event changed nothing in
    the world, which is what a failed attempt looks like from the event log.
    """

    start: int
    end: int
    instruction: str
    succeeded: bool
    observations: tuple[str, ...]


def episode_subtasks(episode) -> tuple[Subtask, ...]:
    """Group an episode's events under the subtask that was in force."""
    bounds: list[tuple[int, str]] = []
    for step, label in enumerate(episode.subtasks):
        if not bounds or bounds[-1][1] != label:
            bounds.append((step, label))

    out: list[Subtask] = []
    for i, (start, label) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(episode)
        seen = tuple(e.text for e in episode.events if start <= e.step < end)
        out.append(Subtask(start, end, label, bool(seen), seen))
    return tuple(out)


@dataclass(frozen=True)
class MemoryUpdate:
    """One low-frequency write, as the high-level policy would emit it.

    `text` is the memory in force *entering* this subtask -- MEM's high-level
    policy emits the memory update together with the subtask it is about to
    pursue, so this is the state it conditions that choice on.
    """

    step: int
    subtask: str
    succeeded: bool
    text: str


@dataclass(frozen=True)
class MemoryTrace:
    texts: tuple[str, ...]            # memory in force at each step
    tokens: tuple[int, ...]           # its size at each step
    beliefs: tuple[Belief, ...]       # for evaluation queries
    updates: tuple[MemoryUpdate, ...]  # the subtask-boundary writes

    def __len__(self) -> int:
        return len(self.texts)


def memory_trace(episode, cfg: MemoryConfig | None = None) -> MemoryTrace:
    """Replay an episode through the memory writer.

    The memory in force at step t reflects everything observed strictly before
    t, which is what the policy would have when choosing action t. Updates are
    emitted at subtask boundaries -- MEM's high-level policy runs at that lower
    frequency, not once per control step.
    """
    cfg = cfg or replace(MemoryConfig(), protected=_protected(episode))
    memory = TextMemory(cfg)
    pending: dict[int, list[Event]] = {}
    for e in episode.events:
        pending.setdefault(e.step, []).append(e)
    # Success is per subtask, not per episode: it is the flag the paper's
    # labeller conditions on, and an episode can succeed with failed attempts
    # inside it.
    outcome = {s.start: s.succeeded for s in episode_subtasks(episode)}

    texts, tokens, beliefs, updates = [], [], [], []
    previous_subtask = None
    for t in range(len(episode)):
        for event in pending.get(t - 1, ()):
            memory.absorb(event)
        text = memory.render()
        texts.append(text)
        tokens.append(cfg.count_tokens(text))
        beliefs.append(memory.snapshot())

        subtask = episode.subtasks[t]
        if subtask != previous_subtask:
            updates.append(
                MemoryUpdate(t, subtask, outcome.get(t, True), text)
            )
            previous_subtask = subtask

    return MemoryTrace(tuple(texts), tuple(tokens), tuple(beliefs), tuple(updates))


def _protected(episode) -> tuple[str, ...]:
    """Object kinds named in the instruction are never dropped for budget."""
    from .world import OBJECT_COLORS

    return tuple(k for k in OBJECT_COLORS if k in episode.instruction)
