"""The VLA-0 input/output contract, engine-independent.

VLA-0 (Goyal et al., 2025, github.com/NVlabs/vla0) is an *unmodified*
Qwen2.5-VL-3B-Instruct fine-tuned to emit robot actions as text: 8 future
timesteps x 7 action dims = 56 integers in [0, 1000], space separated.
Everything an inference engine needs to reproduce its forward pass is here:

  * the exact system prompt and chat layout (`build_prompt`)
  * the image pipeline: two 256x256 LIBERO cameras -> center crop 0.875 ->
    224x224 each -> tiled side by side into one 224x448 image (`tile_obs`)
  * action text <-> continuous action conversion (`parse_action_text`,
    `denormalize`), using the per-dataset min/max stats shipped with the
    checkpoint.

Only numpy + PIL so it imports in every engine's venv.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

HORIZON = 8
ACT_DIM = 7
NUM_BINS = 1000
CROP_FRAC = 0.875            # IMAGE.crop_img in configs/vla0.yaml
IMG_SIZE = 224               # IMAGE.img_size
TILE_W = IMG_SIZE * 2        # two cameras tiled horizontally
N_IMAGE_TOKENS = (IMG_SIZE // 28) * (TILE_W // 28)   # 8 x 16 = 128 after 2x2 merge
MIN_PIXELS = MAX_PIXELS = IMG_SIZE * TILE_W            # 100352: what QwenActor passes to the processor

SYSTEM_MESSAGE = (
    f"Analyze the input image and predict robot actions for the next {HORIZON} timesteps. "
    f"Each action has {ACT_DIM} dimensions. Output a single sequence of {HORIZON * ACT_DIM} "
    f"integers (0-{NUM_BINS} each), representing the {HORIZON} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)

# Qwen2.5-VL chat template, rendered by hand so every engine tokenizes the
# identical string. `Picture 1: ` is what add_vision_id=True inserts.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
EOS_TOKEN = "<|im_end|>"
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643
# token ids for '0'..'9' and ' ' in the Qwen2 tokenizer (QwenActor's NumberSpaceOnlyProcessor)
DIGIT_TOKEN_IDS = list(range(15, 25))
SPACE_TOKEN_ID = 220
ALLOWED_TOKEN_IDS = DIGIT_TOKEN_IDS + [SPACE_TOKEN_ID, EOS_TOKEN_ID]
# 56 numbers, each at most "1000 " = 5 tokens, plus EOS
MAX_NEW_TOKENS = HORIZON * ACT_DIM * 5 + 1


def build_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\nPicture 1: {IMAGE_PLACEHOLDER}{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def chat_messages(instruction: str, image_ref) -> list:
    """OpenAI-style messages (for servers that apply the chat template themselves)."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "image", "image": image_ref},
                                     {"type": "text", "text": instruction}]},
    ]


def _center_crop_resize(img: np.ndarray) -> np.ndarray:
    """256x256 uint8 HWC -> center crop 87.5% -> bilinear resize to 224x224 (torchvision semantics)."""
    h, w = img.shape[:2]
    ch, cw = int(h * CROP_FRAC), int(w * CROP_FRAC)
    top, left = int(round((h - ch) / 2.0)), int(round((w - cw) / 2.0))
    crop = img[top:top + ch, left:left + cw]
    if (ch, cw) != (IMG_SIZE, IMG_SIZE):
        crop = np.asarray(Image.fromarray(crop).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return crop


def tile_obs(agentview: np.ndarray, wrist: np.ndarray) -> Image.Image:
    """Both cameras are already upright (LIBERO renders flipped; prepare_data.py un-flips).
    Returns the single 448x224 PIL image VLA-0 sees (cam order: 3p1=agentview, 3p2=wrist)."""
    a = _center_crop_resize(agentview)
    w = _center_crop_resize(wrist)
    tiled = np.concatenate([a, w], axis=1)
    return Image.fromarray(tiled.astype(np.uint8))


_num_re = re.compile(r"\d+")


def parse_action_text(text: str) -> np.ndarray:
    """Replicates QwenActor.get_action_from_text_action: tolerant to short/long outputs.
    Returns float32 (HORIZON, ACT_DIM) of *bin indices* (0..1000); NaN-free.
    On a hard parse failure returns the midpoint bin everywhere (what VLA-0 does)."""
    ints = [int(x) for x in _num_re.findall(text)]
    if not ints:
        return np.full((HORIZON, ACT_DIM), NUM_BINS / 2.0, dtype=np.float32)
    n = len(ints) - len(ints) % ACT_DIM
    if n == 0:
        return np.full((HORIZON, ACT_DIM), NUM_BINS / 2.0, dtype=np.float32)
    arr = np.asarray(ints[:n], dtype=np.float32).reshape(-1, ACT_DIM)
    if arr.shape[0] < HORIZON:
        arr = np.concatenate([arr, np.repeat(arr[-1:], HORIZON - arr.shape[0], axis=0)], axis=0)
    return arr[:HORIZON]


def is_well_formed(text: str) -> bool:
    toks = text.strip().split()
    return len(toks) == HORIZON * ACT_DIM and all(t.isdigit() and 0 <= int(t) <= NUM_BINS for t in toks)


@dataclass
class ActionStats:
    min: np.ndarray  # (7,)
    max: np.ndarray  # (7,)

    @classmethod
    def load(cls, path: str | Path) -> "ActionStats":
        path = Path(path)
        if path.suffix == ".json":
            d = json.loads(path.read_text())
        else:
            with open(path, "rb") as f:
                d = pickle.load(f)
            d = d["out_ori_act"] if "out_ori_act" in d else d
        return cls(np.asarray(d["min"], dtype=np.float32), np.asarray(d["max"], dtype=np.float32))

    def denormalize(self, bins: np.ndarray) -> np.ndarray:
        return (bins / NUM_BINS) * (self.max - self.min) + self.min

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"min": self.min.tolist(), "max": self.max.tolist()}, indent=2))


def finalize_action(bins: np.ndarray, stats: ActionStats) -> np.ndarray:
    """Continuous (8, 7) action chunk as the LIBERO env consumes it (gripper snapped to +-1)."""
    act = stats.denormalize(bins)
    act[:, -1] = np.where(act[:, -1] > 0, 1.0, -1.0)
    return act


# ----------------------------------------------------------------------------
# Benchmark observation set

@dataclass
class Obs:
    uid: str
    suite: str
    task_id: int
    task_name: str
    instruction: str
    init_state: int
    step: int
    agentview_path: str
    wrist_path: str
    state: Optional[List[float]] = None   # 8-D LIBERO proprio state (eef pos, axis-angle, gripper)

    def tiled(self, root: Path) -> Image.Image:
        a = np.asarray(Image.open(root / self.agentview_path).convert("RGB"))
        w = np.asarray(Image.open(root / self.wrist_path).convert("RGB"))
        return tile_obs(a, w)


def load_manifest(data_dir: str | Path, subset: Optional[str] = None, limit: Optional[int] = None) -> List[Obs]:
    data_dir = Path(data_dir)
    m = json.loads((data_dir / "manifest.json").read_text())
    obs = [Obs(**o) for o in m["observations"]]
    if subset == "core":
        # one init state per task, every suite: 40 observations
        obs = [o for o in obs if o.init_state == m["core_init_state"]]
    if limit:
        obs = obs[:limit]
    return obs
