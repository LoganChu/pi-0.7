"""Shared benchmark loop. Every engine runner implements `infer(batch) -> list[GenOut]`
and hands it to `run_benchmark`, which owns timing, bookkeeping and the result file.

Timing protocol (same for every engine / quantization):
  * warmup: 3 requests at batch 1 (CUDA graphs, JIT, allocator)
  * latency: every observation in the subset, one request at a time, wall-clock
    around the engine call *including* its own preprocessing, excluding our
    PNG decode + tiling (done up front). This is the control-loop number a robot
    sees: observation in -> 56 action integers out.
  * throughput: the same observations sent as one batch of size B for each B
    in --batch-sizes, repeated --repeats times; reports obs/s and output tok/s.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import vla0  # noqa: E402
from common.contracts import get_contract  # noqa: E402


@dataclass
class Request:
    uid: str
    instruction: str
    prompt: str
    images: list                      # PIL images, in placeholder order

    @property
    def image(self):
        return self.images[0]


@dataclass
class GenOut:
    text: str
    n_prompt_tokens: int = -1
    n_out_tokens: int = -1
    ttft_ms: Optional[float] = None   # time to first token when the engine reports it
    extra: dict = field(default_factory=dict)


def gpu_mem_used_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default=None, help="HF-format checkpoint dir (default: the contract's)")
    ap.add_argument("--model-name", default="vla0", choices=["vla0", "vla0-smol"])
    ap.add_argument("--quant", default="bf16")
    ap.add_argument("--engine", default=None, help="label override")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--subset", default="core", choices=["core", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="free-form suffix for the result file")
    ap.add_argument("--notes", default="")


def load_requests(args, contract) -> List[Request]:
    obs = vla0.load_manifest(args.data, subset=args.subset, limit=args.limit)
    reqs = []
    for o in obs:
        prompt, images = contract.build(o, Path(args.data))
        reqs.append(Request(o.uid, o.instruction, prompt, images))
    return reqs


def _stats(xs: Sequence[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    pct = lambda p: xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]  # noqa: E731
    return dict(n=len(xs), mean=statistics.fmean(xs), median=statistics.median(xs),
                p10=pct(10), p90=pct(90), p99=pct(99), min=xs[0], max=xs[-1],
                stdev=statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def run_benchmark(args, engine: str, infer: Callable[[List[Request]], List[GenOut]],
                  reqs: List[Request], env_info: dict, contract, mem_before_load: Optional[int] = None,
                  max_batch: Optional[int] = None) -> dict:
    engine = args.engine or engine
    parse_fn, well_formed_fn = contract.parse, contract.is_well_formed
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    if max_batch:
        batch_sizes = [b for b in batch_sizes if b <= max_batch]
    result = dict(
        model=args.model_name, engine=engine, quant=args.quant, tag=args.tag, notes=args.notes,
        model_path=str(args.model), subset=args.subset, n_obs=len(reqs),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(), env=env_info,
        gpu_mem_used_mib_before_load=mem_before_load, gpu_mem_used_mib_after_load=gpu_mem_used_mib(),
    )

    # --- warmup
    for i in range(args.warmup):
        infer([reqs[i % len(reqs)]])

    # --- latency, batch 1
    outputs, lat = {}, []
    if not args.no_latency:
        for r in reqs:
            t0 = time.perf_counter()
            o = infer([r])[0]
            dt = (time.perf_counter() - t0) * 1000
            lat.append(dt)
            outputs[r.uid] = o
            print(f"  {r.uid:28s} {dt:8.1f} ms  out_tok={o.n_out_tokens:4d}  {o.text[:60]!r}", flush=True)
        result["latency_ms"] = _stats(lat)
        toks = [outputs[r.uid].n_out_tokens for r in reqs if outputs[r.uid].n_out_tokens > 0]
        if toks:
            result["out_tokens"] = _stats(toks)
            result["decode_tok_per_s_batch1"] = statistics.fmean(
                outputs[r.uid].n_out_tokens / (d / 1000) for r, d in zip(reqs, lat) if outputs[r.uid].n_out_tokens > 0)
        ttfts = [outputs[r.uid].ttft_ms for r in reqs if outputs[r.uid].ttft_ms is not None]
        if ttfts:
            result["ttft_ms"] = _stats(ttfts)
        result["well_formed_rate"] = float(np.mean([well_formed_fn(outputs[r.uid].text) for r in reqs]))
        result["predictions"] = {
            r.uid: dict(text=outputs[r.uid].text, bins=parse_fn(outputs[r.uid].text).astype(int).tolist(),
                        n_out_tokens=outputs[r.uid].n_out_tokens, n_prompt_tokens=outputs[r.uid].n_prompt_tokens,
                        extra=outputs[r.uid].extra)
            for r in reqs}

    # --- throughput at larger batch sizes
    thr = {}
    for b in batch_sizes:
        if b == 1 and not args.no_latency:
            thr["1"] = dict(obs_per_s=1000.0 / result["latency_ms"]["mean"],
                            out_tok_per_s=result.get("decode_tok_per_s_batch1"))
            continue
        batch = (reqs * ((b + len(reqs) - 1) // len(reqs)))[:b]
        times, ntoks = [], []
        infer(batch[: min(b, 4)])  # shape warmup
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            outs = infer(batch)
            times.append(time.perf_counter() - t0)
            ntoks.append(sum(max(o.n_out_tokens, 0) for o in outs))
        best = min(times)
        thr[str(b)] = dict(batch=b, wall_s=best, obs_per_s=b / best,
                           out_tok_per_s=(ntoks[times.index(best)] / best) if ntoks[times.index(best)] > 0 else None,
                           repeats=args.repeats)
        print(f"  batch {b:3d}: {b / best:7.2f} obs/s  {thr[str(b)]['out_tok_per_s'] or 0:8.1f} out tok/s", flush=True)
    result["throughput"] = thr
    result["gpu_mem_used_mib_peak_seen"] = gpu_mem_used_mib()

    out = args.out or str(ROOT / "results" / f"{args.model_name}__{engine}__{args.quant}{('__' + args.tag) if args.tag else ''}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1))
    lm = result.get("latency_ms", {})
    print(f"\n== {args.model_name} / {engine} / {args.quant}: latency mean {lm.get('mean', float('nan')):.1f} ms "
          f"(p90 {lm.get('p90', float('nan')):.1f}), well-formed {result.get('well_formed_rate', float('nan')):.2f} -> {out}")
    return result
