"""The VLA-0-Smol input/output contract (Robot-Learning-Collective/VLA-0-Smol, LIBERO).

SmolVLM2-500M-Video-Instruct fine-tuned (LeRobot policy `vla0_smol`) to emit
8 timesteps x 7 dims = 56 integers in [0, 512) as text. Unlike VLA-0 it also
reads the 8-D proprio state, binned the same way, and feeds the two cameras as
two separate images (center-cropped 230x230, resized by the processor to the
512x512 SigLIP input -> 64 tokens each).

Reproduced from lerobot-experiments/src/lerobot/policies/vla0_smol/modeling_vla0_smol.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

HORIZON = 8
ACT_DIM = 7
STATE_DIM = 8
N_BINS = 512
EPS = 1e-6
CROP = 230
MAX_NEW_TOKENS = 512          # config.max_decoding_steps; 56 x (<=3 digits + space) + EOS in practice
EOS_TOKEN_ID = 49279          # <end_of_utterance>
PAD_TOKEN_ID = 2
IMAGE_TOKEN_ID = 49190
IMAGE_SEQ_LEN = 64

_edges = np.linspace(-1.0 - EPS, 1.0 + EPS, N_BINS + 1, dtype=np.float64)
_bins = _edges[:-1]
_centers = 0.5 * (_edges[:-1] + _edges[1:])


@dataclass
class Stats:
    action_min: np.ndarray
    action_max: np.ndarray
    state_min: np.ndarray
    state_max: np.ndarray

    @classmethod
    def load(cls, path) -> "Stats":
        d = json.loads(Path(path).read_text())
        f = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
        return cls(f(d["action"]["min"]), f(d["action"]["max"]), f(d["state"]["min"]), f(d["state"]["max"]))


def normalize_minmax(x, lo, hi):
    # LeRobot NormalizerProcessorStep MIN_MAX: (x - min) / (max - min + eps) * 2 - 1
    return (x - lo) / (hi - lo + 1e-8) * 2.0 - 1.0


def discretize(x_norm: np.ndarray) -> np.ndarray:
    # torch.bucketize(x, bins) - 1 with bins = edges[:-1]
    return np.searchsorted(_bins, x_norm, side="left") - 1


def state_text(state: np.ndarray, stats: Stats) -> str:
    s = discretize(normalize_minmax(np.asarray(state, dtype=np.float64), stats.state_min, stats.state_max))
    return " ".join(str(int(v)) for v in s)


def prefix_text(instruction: str, state, stats: Stats) -> str:
    task = instruction.lower().strip().replace("_", " ")
    return f"Task: {task}, State: {state_text(state, stats)}, Actions: "


def build_prompt(instruction: str, state, stats: Stats) -> str:
    """SmolVLM chat template with two <image> placeholders then the text; add_generation_prompt."""
    return f"<|im_start|>User:<image><image>{prefix_text(instruction, state, stats)}<end_of_utterance>\nAssistant:"


def center_crop(img: np.ndarray) -> Image.Image:
    h, w = img.shape[:2]
    top, left = int(round((h - CROP) / 2.0)), int(round((w - CROP) / 2.0))
    return Image.fromarray(np.ascontiguousarray(img[top:top + CROP, left:left + CROP]))


def images_for(obs, root: Path) -> List[Image.Image]:
    a = np.asarray(Image.open(root / obs.agentview_path).convert("RGB"))
    w = np.asarray(Image.open(root / obs.wrist_path).convert("RGB"))
    return [center_crop(a), center_crop(w)]


_num_re = re.compile(r"\d+")


def is_well_formed(text: str) -> bool:
    toks = text.strip().split()
    return len(toks) == HORIZON * ACT_DIM and all(t.isdigit() and 0 <= int(t) < N_BINS for t in toks)


def parse_action_text(text: str) -> np.ndarray:
    """Bin indices (8, 7). The reference implementation returns all-zero bins on any malformed
    output; we keep that behaviour for exact fidelity accounting."""
    toks = text.strip().split()
    if not is_well_formed(text):
        return np.zeros((HORIZON, ACT_DIM), dtype=np.float32)
    return np.asarray([int(t) for t in toks], dtype=np.float32).reshape(HORIZON, ACT_DIM)


def denormalize(bins: np.ndarray, stats: Stats) -> np.ndarray:
    c = _centers[np.clip(bins.astype(int), 0, N_BINS - 1)]
    # LeRobot UnnormalizerProcessorStep MIN_MAX: (x + 1) / 2 * (max - min + eps) + min
    return ((c + 1.0) / 2.0 * (stats.action_max - stats.action_min + 1e-8) + stats.action_min).astype(np.float32)
