"""One object per benchmarked VLA that tells an engine runner how to build the
prompt, which images to attach, and how to turn generated text back into actions."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from common import vla0, vla0_smol

ROOT = Path(__file__).resolve().parents[1]


class Contract:
    name: str
    model_dir: Path
    n_images: int
    max_new_tokens: int
    eos_token_id: int
    pad_token_id: int
    allowed_token_ids: Optional[List[int]] = None
    mm_processor_kwargs: dict = {}
    hf_auto_class: str = "AutoModelForImageTextToText"

    def build(self, obs, root: Path):  # -> (prompt, [PIL images])
        raise NotImplementedError

    def parse(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def is_well_formed(self, text: str) -> bool:
        raise NotImplementedError

    def denormalize(self, bins: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class VLA0Contract(Contract):
    name = "vla0"
    model_dir = ROOT / "models/vla0-libero-hf"
    n_images = 1
    max_new_tokens = vla0.MAX_NEW_TOKENS
    eos_token_id = vla0.EOS_TOKEN_ID
    pad_token_id = vla0.PAD_TOKEN_ID
    allowed_token_ids = vla0.ALLOWED_TOKEN_IDS
    mm_processor_kwargs = {"min_pixels": vla0.MIN_PIXELS, "max_pixels": vla0.MAX_PIXELS}
    hf_auto_class = "Qwen2_5_VLForConditionalGeneration"

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else self.model_dir
        p = self.model_dir / "action_stats.json"
        self.stats = vla0.ActionStats.load(p) if p.exists() else None

    def build(self, obs, root):
        return vla0.build_prompt(obs.instruction), [obs.tiled(root)]

    def parse(self, text):
        return vla0.parse_action_text(text)

    def is_well_formed(self, text):
        return vla0.is_well_formed(text)

    def denormalize(self, bins):
        return self.stats.denormalize(bins)


class VLA0SmolContract(Contract):
    name = "vla0-smol"
    model_dir = ROOT / "models/vla0-smol-libero-hf"
    n_images = 2
    max_new_tokens = vla0_smol.MAX_NEW_TOKENS
    eos_token_id = vla0_smol.EOS_TOKEN_ID
    pad_token_id = vla0_smol.PAD_TOKEN_ID
    allowed_token_ids = None
    mm_processor_kwargs = {"do_image_splitting": False}

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else self.model_dir
        self.stats = vla0_smol.Stats.load(self.model_dir / "action_stats.json")

    def build(self, obs, root):
        assert obs.state is not None, "VLA-0-Smol needs the proprio state; re-run prepare_data.sh"
        return vla0_smol.build_prompt(obs.instruction, np.asarray(obs.state), self.stats), vla0_smol.images_for(obs, root)

    def parse(self, text):
        return vla0_smol.parse_action_text(text)

    def is_well_formed(self, text):
        return vla0_smol.is_well_formed(text)

    def denormalize(self, bins):
        return vla0_smol.denormalize(bins, self.stats)


def get_contract(name: str, model_dir=None) -> Contract:
    return {"vla0": VLA0Contract, "vla0-smol": VLA0SmolContract}[name](model_dir)
