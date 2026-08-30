"""memlab -- MEM's memory architecture, shrunk to a 2-D kitchen.

MEM (Multi-Scale Embodied Memory) gives a VLA two memories at different time
scales: a compressed *video* memory for recent detail, and a *text* memory of
summarized events for the long horizon. It is built on pi-0.6, and like pi-0.7
it ships as a paper with no weights and no code.

What is reproducible on a laptop is the part that is purely architectural. This
package rebuilds the short-horizon video encoder exactly -- factorized
space-time attention, history dropped in the upper layers, no new parameters --
and puts it in a 2-D partially observable kitchen where memory is load-bearing,
so its claims can be measured rather than taken on faith.

    world.py    the kitchen: continuous motion, cabinets that hide things
    tasks.py    task generators built so history is required to act correctly
    oracle.py   scripted demonstrator, deliberately searchy
    encoder.py  the token-dropping space-time ViT
    cost.py     analytic FLOP model for the three cost regimes
    render.py   watch an episode: GIF playback and the figure helpers
    text_memory.py      the long-horizon half: compressed text memory
    memory_tokenizer.py word-level vocabulary over that memory
    llm_memory.py       the same writer, but a real LM does the writing

To watch the oracle run:

    python -m memlab.render --task find_object --seed 0

Actions are continuous (dx, dy, grip), which is what lets `fastlab`'s FAST
tokenizer consume them: the same chunk -> DCT -> BPE path, on 3 dimensions
instead of 14.

Reference: Torne et al., "MEM: Multi-Scale Embodied Memory for Vision Language
Action Models" (2026). https://arxiv.org/abs/2603.03596
"""

from .cost import Cost, flops, joint_flops, tokens_per_layer
from .encoder import EncoderConfig, MemEncoder
from .oracle import Episode, collect, run_episode
from .render import rollout, save_gif, visibility
from .memory_tokenizer import MemoryTokenizer
from .tasks import TaskSpec, make_task
from .text_memory import (
    Belief,
    MemoryConfig,
    MemoryTrace,
    MemoryUpdate,
    Subtask,
    TextMemory,
    episode_subtasks,
    memory_trace,
)
from .world import Kitchen, Layout, Obj, default_layout

__all__ = [
    "Kitchen",
    "Layout",
    "Obj",
    "default_layout",
    "TaskSpec",
    "make_task",
    "Episode",
    "collect",
    "run_episode",
    "rollout",
    "save_gif",
    "visibility",
    "TextMemory",
    "MemoryConfig",
    "MemoryTrace",
    "MemoryUpdate",
    "Belief",
    "Subtask",
    "memory_trace",
    "episode_subtasks",
    "MemoryTokenizer",
    "MemEncoder",
    "EncoderConfig",
    "Cost",
    "flops",
    "joint_flops",
    "tokens_per_layer",
]

__version__ = "0.1.0"
