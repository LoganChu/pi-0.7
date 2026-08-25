# fastlab — understanding the π₀.x pieces you can run on a laptop

Transparent, from-scratch reimplementations of two components of Physical
Intelligence's π₀.x models. The goal is *understanding*, not speed: every stage
is a separate, inspectable function, and the experiment scripts turn each one
into a figure.

**`fastlab`** — **FAST** (Frequency-space Action Sequence Tokenization), the
action tokenizer behind π₀-FAST and the autoregressive half of the later π₀.x
models.

```
raw actions → quantile normalize → DCT-II over time → quantize
            → flatten → drop trailing zeros → BPE → tokens
```

**`memlab`** — the short-horizon video encoder from **MEM** (Multi-Scale
Embodied Memory), the memory architecture built on π₀.6, together with a 2-D
partially observable kitchen where memory is load-bearing.

```
frame history → patch embed → [spatial attn every layer, temporal every 4th]
              → drop history above layer 6 → current-frame tokens
```

## Why this repo exists

π₀.7 itself is **not open** — Physical Intelligence published it (April 2026) as
a paper and a product, with no weights or code. What *is* open is the machinery
it is built on, and FAST is the piece you can fully reproduce on a laptop with
no GPU at all. FAST is pure numpy/scipy plus a BPE vocabulary; the entire lab
runs on CPU in seconds per experiment.

| π₀.x technique | Status here |
|---|---|
| FAST action tokenization | ✅ reimplemented + validated against the official tokenizer |
| MEM short-horizon video memory | ✅ architecture reimplemented; measured, not trained |
| MEM long-horizon text memory | ⬜ next — see "Where this goes" below |
| Flow-matching action expert | ➡️ use [`openpi`](https://github.com/Physical-Intelligence/openpi) (π₀ / π₀.5), needs a real GPU |
| Knowledge insulation | ➡️ π₀.5 in `openpi` |
| Multimodal prompting, world-model subgoals | ❌ not released |

MEM is a separate 2026 paper built *on* π₀.6, not part of the π₀.6 release
itself — π\*0.6 was the RL-from-experience work (Recap). Like π₀.7 it ships with
no weights and no code, so `memlab` is a reimplementation from the paper, never
a reproduction of its results.

## Setup

```bash
uv sync                    # core lab: numpy, scipy, matplotlib, tokenizers
uv sync --extra data       # + real robot action chunks from LeRobot datasets
uv sync --extra official   # + transformers/torch, for the parity check
uv sync --extra mem        # + torch, for the memlab experiments (07-09)
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
uv run python experiments/07_token_dropping_cost.py   # needs --extra mem
uv run python experiments/08_temporal_stride.py       # needs --extra mem
uv run python experiments/09_oracle_rollout.py        # needs --extra mem

# watch an episode play out, as an animated GIF
uv run python -m memlab.render --task find_object --seed 0
uv run python -m memlab.render --task tidy_up --seed 3 --stride 3
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
| `07_token_dropping_cost` | What does dropping history in the upper layers actually buy? |
| `08_temporal_stride` | How often does temporal attention need to run, and how far back can it reach? |
| `09_oracle_rollout` | What does one episode look like, and which facts are only in memory? |

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

### MEM's memory encoder

At 2-D scale — 64×64 frames, 64 patches, dim 192, depth 12, history cut at
layer 6 — three of the paper's claims reproduce, and one thing it does not
mention falls out of the design:

**"No new learnable parameters" is exact, and it pins the architecture.** The
encoder is **5,388,096** parameters with temporal attention and **5,388,096**
without: bitwise the same count as a single-image ViT. That only holds because
temporal attention reuses the spatial block's weights and norm, and time is
encoded sinusoidally. A separately parameterized temporal block would cost
**7,171,392** (+33%). The paper states the claim but not the mechanism, so this
is a *reading* of it, flagged as such in `encoder.py`.

**Dropping history buys a slope, not a ceiling.** At 18 frames of history:

| | GFLOPs @ T=18 | vs 1 frame |
|---|---|---|
| joint attention over all T×N tokens | 24.55 | 34× |
| factorized, keep history | 14.06 | 19× |
| MEM (factorized + drop) | 7.25 | 10× |

Exactly half the slope of keeping history, because half the depth sits above the
cut. On an unloaded laptop the measured wall-clock ratio tracks that closely —
10.2× across T=1→18 against 10.0× predicted. Absolute times swing several-fold
with background load, which is why panel (c) is the sanity check and the FLOP
count is the claim. `cost.measure_grid` interleaves conditions for that reason;
timing them one at a time let a background spike land entirely on one and
reported T=12 as slower than T=18.

**Two of the three temporal layers do nothing.** With temporal attention every
4th layer (3, 7, 11) and history dropped above layer 6, only layer 3 has a
history left to attend over. `temporal_every` and `drop_history_after` cannot be
chosen independently, which the paper does not say — at stride 12 the encoder
has *no* working temporal path at all despite nominally having one.

**Where temporal mixing sits matters as much as how often.** Strides 4 and 6
cost identically (4.5% of the forward pass, one working layer each), yet
perturbing a past frame moves the output ~29% more at stride 4 — mixing earlier
leaves more full-history depth above it to propagate through. Measured at random
initialization, so this is the architecture's *capacity* to carry information
across time, not evidence about what a trained model would use. The control is
what makes it readable: with temporal attention off, past-frame sensitivity is
exactly zero, because nothing else in the encoder mixes across time.

### Where memory is actually needed

Experiment 09 plots, per step, whether the fact a policy would need is visible
in the agent's own view. It turned up something uncomfortable about
`find_object`:

- *The target is on screen for 62% of the episode.* The oracle opens the cabinet
  and picks the bread up two steps later, then carries it — and a held object is
  drawn on the agent. So "remember where the bread is" is barely a memory demand
  at all.
- *The real demand is "which cabinets have I already searched".* That fact is
  encoded in the open doors, which leave the view at step 55 and never come
  back. For the last 44 steps it exists only in memory.
- *`tidy_up` is the stronger memory task, and not by degree.* The agent closes
  each door behind it, so a used cabinet is pixel-identical to an unused one.
  Progress is not recoverable from the observation at *any* step, and the target
  object is on screen for only 31% of the episode.

Worth knowing before building the ablation ladder: a memory ablation run on
`find_object` alone would understate how much memory matters.

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
| `memlab/world.py` | 2-D kitchen: continuous motion, cabinets that hide things, reach geometry |
| `memlab/tasks.py` | task generators built so history is required to act correctly |
| `memlab/oracle.py` | scripted demonstrator, deliberately searchy |
| `memlab/encoder.py` | the token-dropping space-time ViT |
| `memlab/cost.py` | analytic FLOP model for the three cost regimes |
| `memlab/render.py` | GIF playback of an episode + figure helpers |

`memlab` actions are continuous `(dx, dy, grip)`, which is what lets `fastlab`
consume them: the same chunk → DCT → BPE path, on 3 dimensions instead of 14.
The gripper channel is near-binary, which `fastlab.data.gripper_like_chunks`
already flags as the pathological case for DCT.

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

## Where this goes

`memlab` currently stops before training, which is the honest boundary: nothing
above required a single gradient step, and every claim it makes is about the
architecture rather than about a model's competence. The next pieces, roughly in
order of how much they need:

1. **Text memory** — MEM's long-horizon half. The kitchen already emits an event
   log (`opened cabinet 3`, `saw bread in cabinet 3`) in exactly the form the
   summarization stage consumes. Templates first, then a small pretrained LM in
   the role the paper gives an "off-the-shelf LLM".
2. **The hierarchical policy and a training loop** — high-level subtask labels
   are already recorded per step; the low-level half consumes FAST tokens.
   Wants a CUDA build of torch, not the CPU one these experiments run on.
3. **The ablation ladder** — no memory / proprioceptive-only / pooled /
   short-term only / long-term only / full MEM, on success rate.
4. **The modality crossover** — vary the delay between seeing an object and
   needing it, and find the horizon where text memory overtakes video memory.
   That curve is the paper's whole thesis and it does not plot it.
5. **Attention over memory** — which past frame, which text entry drove this
   action. The paper has no interpretability section at all, so there is nothing
   to reproduce here, only something to build.

### What a 2-D lab cannot tell you

No VLM pretraining, so a from-scratch model here is a specialist and any
generalization claim is out of reach. No dexterity, so the manipulation
difficulty that makes MEM's real tasks hard is absent. Fifteen minutes of
wall-clock becomes N steps — the step count matches, the real-time drift, sensor
noise and compounding physical error do not. And 64 patches per frame is far
fewer than MEM works with, which flatters the joint-attention baseline in
experiment 07 (its penalty is quadratic in exactly that number).

## References

- [FAST: Efficient Action Tokenization for VLA Models](https://arxiv.org/abs/2501.09747) — Pertsch et al., 2025
- [FAST project page](https://www.pi.website/research/fast)
- [`physical-intelligence/fast`](https://huggingface.co/physical-intelligence/fast) — official universal tokenizer
- [`openpi`](https://github.com/Physical-Intelligence/openpi) — π₀, π₀-FAST, π₀.5 weights and code
- [MEM: Multi-Scale Embodied Memory for VLA Models](https://arxiv.org/abs/2603.03596) — Torne et al., 2026
- [MEM project page](https://www.pi.website/research/memory) — task videos and the live text memory
- [π\*0.6: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759) — Recap, the actual π₀.6 release ([blog](https://www.pi.website/blog/pistar06))
