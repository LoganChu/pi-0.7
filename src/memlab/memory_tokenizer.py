"""Tokenizing the text memory, so its cost is measured in the units that matter.

Text memory competes with everything else for context, so the honest unit is
tokens, not words. This is the text-side counterpart to `fastlab.bpe`: the same
question -- what does one more thing to remember cost -- asked of language
instead of action coefficients.

WordLevel rather than BPE, deliberately. The memory language is closed: the
template writer emits a fixed set of words plus cabinet indices, so a word-level
vocabulary is exact and every id is interpretable. Swap in BPE when the writer
becomes a real LM (`llm_memory.py`) and starts producing open vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

UNK = "[UNK]"


class MemoryTokenizer:
    """A word-level vocabulary over rendered memory text."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def fit(cls, texts: Sequence[str], vocab_size: int = 512) -> "MemoryTokenizer":
        tokenizer = Tokenizer(models.WordLevel(unk_token=UNK))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size, special_tokens=[UNK], min_frequency=1
        )
        tokenizer.train_from_iterator(list(texts), trainer)
        return cls(tokenizer)

    @classmethod
    def from_episodes(cls, tasks: Iterable[str] = ("find_object", "tidy_up"),
                      n_episodes: int = 12, seed: int = 0) -> "MemoryTokenizer":
        """Fit on memories generated from real episodes of each task."""
        from .oracle import collect
        from .text_memory import MemoryConfig, memory_trace

        corpus: list[str] = []
        for task in tasks:
            for episode in collect(task, n_episodes, seed=seed, keep_frames=False):
                corpus.extend(memory_trace(episode).texts)
                corpus.extend(memory_trace(episode, MemoryConfig(compact=False)).texts)
        return cls.fit(corpus)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def count(self, text: str) -> int:
        """Token count -- pass this as `MemoryConfig.count_tokens`."""
        return len(self._tok.encode(text).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def unknown_rate(self, text: str) -> float:
        """Fraction of tokens that fell out of vocabulary.

        Worth checking after any change to the writer: a nonzero rate means the
        vocabulary was fit on a corpus that no longer covers what it emits.
        """
        ids = self.encode(text)
        if not ids:
            return 0.0
        unk = self._tok.token_to_id(UNK)
        return sum(i == unk for i in ids) / len(ids)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(out))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "MemoryTokenizer":
        return cls(Tokenizer.from_file(str(path)))
