# fastlab — understanding FAST action tokenization

A transparent, from-scratch reimplementation of **FAST** (Frequency-space Action
Sequence Tokenization), the action tokenizer behind π₀-FAST and the
autoregressive half of Physical Intelligence's later π₀.x models.

The goal is *understanding*, not speed: every stage of the pipeline is a
separate, inspectable function, and the experiment scripts turn each one into a
figure.

```
raw actions → quantile normalize → DCT-II over time → quantize
            → flatten → drop trailing zeros → BPE → tokens
```

## Why this repo exists

π₀.7 itself is **not open** — Physical Intelligence published it (April 2026) as
a paper and a product, with no weights or code. What *is* open is the machinery
it is built on, and FAST is the piece you can fully reproduce on a laptop with
no GPU at all. FAST is pure numpy/scipy plus a BPE vocabulary; the entire lab
runs on CPU in seconds per experiment.

| π₀.7 technique | Status here |
|---|---|
| FAST action tokenization | ✅ reimplemented + validated against the official tokenizer |
| Flow-matching action expert | ➡️ use [`openpi`](https://github.com/Physical-Intelligence/openpi) (π₀ / π₀.5), needs a real GPU |
| Knowledge insulation | ➡️ π₀.5 in `openpi` |
| Multimodal prompting, world-model subgoals | ❌ not released |

## Setup

```bash
uv sync                    # core lab: numpy, scipy, matplotlib, tokenizers
uv sync --extra data       # + real robot action chunks from LeRobot datasets
uv sync --extra official   # + transformers/torch, for the parity check
```

No CUDA, no WSL, no cluster. Runs natively on Windows.

## Run the experiments

```bash
uv run python experiments/01_pipeline_walkthrough.py
uv run python experiments/02_flatten_order.py
uv run python experiments/03_rate_distortion.py
uv run python experiments/04_control_frequency.py
uv run python experiments/05_bpe_anatomy.py
uv run python experiments/06_official_parity.py   # needs --extra official
```

Figures land in `figures/`.

| Script | Question it answers |
|---|---|
| `01_pipeline_walkthrough` | What does each stage actually do to one action chunk? |
| `02_flatten_order` | Does frequency-major vs dimension-major flattening matter? ([openpi#306](https://github.com/Physical-Intelligence/openpi/issues/306)) |
| `03_rate_distortion` | What does the `scale` knob buy — tokens vs reconstruction error? |
| `04_control_frequency` | Why does FAST beat naive binning by *more* at higher control rates? |
| `05_bpe_anatomy` | What did BPE actually learn to merge? |
| `06_official_parity` | Does this reimplementation match `physical-intelligence/fast`? |

## What the experiments found

On synthetic band-limited chunks (50 steps × 14 dims, 50 Hz, `scale=10`,
vocab 1024):

**Validated against the real tokenizer.** fastlab and
`physical-intelligence/fast` produce **bitwise identical reconstructions**
(max difference exactly `0.0`) at comparable token counts (94.1 vs 92.5). Token
*ids* differ, as they must — different BPE vocabularies — so the DCT,
quantization and inverse are confirmed to match the official implementation.

**FAST's advantage scales with control frequency**, which is the paper's central
claim, reproduced here end to end:

| control rate | FAST tokens | naive tokens | advantage | coefficients zeroed |
|---|---|---|---|---|
| 5 Hz | 283 | 700 | 2.5× | 36% |
| 50 Hz | 88 | 700 | 7.9× | 82% |
| 200 Hz | 44 | 700 | 15.8× | 91% |

**Compression breaks down as** ~3.1× from dropping trailing zeros and a further
~2.5× from BPE, for ~7.9× overall. BPE lands *below* the raw non-zero
coefficient count (89 tokens vs 127 non-zeros), because a merged token can stand
for several non-zero coefficients at once.

**Two findings worth knowing:**

- *Flatten order behaves counter-intuitively.* Frequency-major trims ~3× shorter
  before BPE (225 vs 661 coefficients) — yet dimension-major still ends up
  ~1.10× cheaper in final tokens, because BPE merges its repeated per-dimension
  zero runs. Pre-BPE stream length is a misleading proxy for token cost.
- *The accuracy knob has a hard ceiling.* Past `scale ≈ 70` the quantized
  integers span more distinct values than a 1024 vocabulary can hold, leaving no
  room for merges. Vocabulary size, not precision, is what bounds `scale`.

Regenerate everything with the commands above; figures land in `figures/`.

## Library

```python
import numpy as np
from fastlab import FastTokenizer, FastConfig
from fastlab.data import synthetic_chunks

chunks = synthetic_chunks(n_chunks=2000, horizon=50, action_dim=14, control_hz=50)

tok = FastTokenizer.fit(chunks, FastConfig(scale=10.0, vocab_size=1024))
tokens = tok.encode(chunks[0])          # -> list[int]
recon  = tok.decode(tokens)             # -> (50, 14)

trace = tok.encode_trace(chunks[0])     # every intermediate, for plotting
trace.coefficients, trace.quantized, trace.stream, trace.sparsity
```

`FastTokenizer.fit` reproduces FAST's per-dataset training path. The published
`physical-intelligence/fast` checkpoint is the *universal* tokenizer, fit across
many robot embodiments.

## Module map

| Module | Contents |
|---|---|
| `fastlab/normalize.py` | per-dimension quantile normalization |
| `fastlab/dct.py` | DCT-II, quantization, flattening, zero-trimming |
| `fastlab/bpe.py` | integer↔char codec + BPE training over coefficient streams |
| `fastlab/tokenizer.py` | the assembled pipeline, with `encode_trace` |
| `fastlab/baselines.py` | naive per-timestep binning (RT-1 / OpenVLA style) |
| `fastlab/metrics.py` | reconstruction error, compression stats |
| `fastlab/data.py` | synthetic band-limited chunks + LeRobot dataset loader |
| `fastlab/viz.py` | shared plotting style and validated palette |

## Scaling up beyond the tokenizer

This lab deliberately stops at the tokenizer, because that is where the CPU-only
work ends. For anything that trains a policy:

| Task | Memory | Where |
|---|---|---|
| π₀ inference | > 8 GB | marginal on a 12 GB laptop GPU |
| LoRA fine-tune | > 22.5 GB | cluster / cloud |
| Full fine-tune | > 70 GB | A100 80 GB / H100 |

Note that `openpi` is JAX-first and Ubuntu-only, and its PyTorch port does *not*
support π₀-FAST. [LeRobot's `pi0fast`](https://huggingface.co/docs/lerobot/pi0fast)
is the gentler path if you want a policy running locally.

## References

- [FAST: Efficient Action Tokenization for VLA Models](https://arxiv.org/abs/2501.09747) — Pertsch et al., 2025
- [FAST project page](https://www.pi.website/research/fast)
- [`physical-intelligence/fast`](https://huggingface.co/physical-intelligence/fast) — official universal tokenizer
- [`openpi`](https://github.com/Physical-Intelligence/openpi) — π₀, π₀-FAST, π₀.5 weights and code
