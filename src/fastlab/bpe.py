"""Byte-pair encoding over streams of quantized DCT coefficients.

FAST's final stage trains a BPE vocabulary over the integer coefficient stream.
BPE is what converts the long runs of zeros (and the repeated low-magnitude
patterns) left by quantization into single dense tokens -- this is where the
order-of-magnitude compression over naive binning actually comes from.

`tokenizers` operates on text, so integers are first mapped to single characters
by `IntCodec`. The mapping is order-preserving and injective, so BPE sees
exactly the structure of the integer stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tokenizers import Tokenizer, models, trainers

UNK_TOKEN = "<unk>"

# Characters are allocated starting here. 0x100 clears ASCII and the C1 control
# block; the alphabet stays far below the UTF-16 surrogate range at 0xD800.
_CHAR_BASE = 0x100
_MAX_ALPHABET = 0xD000 - _CHAR_BASE


@dataclass(frozen=True)
class IntCodec:
    """Order-preserving map between a bounded integer range and single chars."""

    lo: int
    hi: int

    @classmethod
    def fit(cls, streams: Iterable[np.ndarray], margin: int = 8) -> "IntCodec":
        """Bound the codec by the observed integer range, plus a little headroom.

        The margin lets chunks slightly outside the training distribution encode
        without clipping. Anything beyond it is clamped -- `encode` reports how
        often that happens so it does not fail silently.
        """
        lo, hi = 0, 0
        for stream in streams:
            arr = np.asarray(stream)
            if arr.size == 0:
                continue
            lo = min(lo, int(arr.min()))
            hi = max(hi, int(arr.max()))
        lo, hi = lo - margin, hi + margin
        if (hi - lo + 1) > _MAX_ALPHABET:
            raise ValueError(
                f"integer range [{lo}, {hi}] exceeds the {_MAX_ALPHABET}-symbol "
                "alphabet limit; lower the quantization scale"
            )
        return cls(lo=lo, hi=hi)

    @property
    def size(self) -> int:
        return self.hi - self.lo + 1

    @property
    def alphabet(self) -> list[str]:
        return [chr(_CHAR_BASE + i) for i in range(self.size)]

    def encode(self, stream: np.ndarray) -> tuple[str, int]:
        """Map an integer stream to a string. Returns (text, n_clipped)."""
        arr = np.asarray(stream, dtype=np.int64).reshape(-1)
        clipped = np.clip(arr, self.lo, self.hi)
        n_clipped = int(np.count_nonzero(clipped != arr))
        return "".join(chr(_CHAR_BASE + int(v) - self.lo) for v in clipped), n_clipped

    def decode(self, text: str) -> np.ndarray:
        out = [ord(ch) - _CHAR_BASE + self.lo for ch in text]
        return np.asarray(out, dtype=np.int32)


class BPECodec:
    """A BPE vocabulary trained over integer coefficient streams."""

    def __init__(self, tokenizer: Tokenizer, codec: IntCodec):
        self._tokenizer = tokenizer
        self._codec = codec

    @classmethod
    def train(
        cls,
        streams: Sequence[np.ndarray],
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> "BPECodec":
        codec = IntCodec.fit(streams)
        alphabet = codec.alphabet
        if vocab_size <= len(alphabet):
            raise ValueError(
                f"vocab_size={vocab_size} must exceed the {len(alphabet)}-symbol "
                "base alphabet, otherwise there is no room for any merges"
            )

        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        # No pre-tokenizer on purpose: merges must be free to span the whole
        # coefficient stream rather than stopping at word boundaries.
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[UNK_TOKEN],
            initial_alphabet=alphabet,
            show_progress=False,
        )
        texts = [codec.encode(s)[0] for s in streams]
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tokenizer, codec)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def codec(self) -> IntCodec:
        return self._codec

    def encode(self, stream: np.ndarray) -> list[int]:
        text, _ = self._codec.encode(stream)
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> np.ndarray:
        pieces = []
        for i in ids:
            token = self._tokenizer.id_to_token(int(i))
            # Unknown/special ids carry no coefficients; skipping them keeps a
            # malformed generation decodable instead of raising.
            if token is None or token == UNK_TOKEN:
                continue
            pieces.append(token)
        return self._codec.decode("".join(pieces))

    def merge_lengths(self) -> dict[int, int]:
        """Token id -> number of raw coefficients it expands to.

        A histogram over this is the clearest single view of what BPE bought
        you: a vocabulary that is mostly length-1 learned nothing.
        """
        vocab = self._tokenizer.get_vocab()
        return {i: len(t) for t, i in vocab.items() if t != UNK_TOKEN}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))
        (path / "codec.json").write_text(
            json.dumps({"lo": self._codec.lo, "hi": self._codec.hi}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPECodec":
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        meta = json.loads((path / "codec.json").read_text(encoding="utf-8"))
        return cls(tokenizer, IntCodec(lo=meta["lo"], hi=meta["hi"]))
