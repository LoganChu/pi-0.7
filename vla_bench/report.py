"""Aggregate results/*.json into one table + fidelity-vs-reference metrics.

Fidelity is computed per model against a reference run (default: the engine's own
bf16 run if present, else hf/bf16), on the observations both runs share:
  exact_match   : fraction of observations whose 56-integer action text is identical
  mae_bins      : mean |pred - ref| in bin units (0..1000)  -> /1000 = fraction of range
  mae_action    : mean |pred - ref| after de-normalization, in the env's action units,
                  averaged over the 6 continuous dims (gripper excluded)
  gripper_agree : fraction of timesteps whose gripper sign matches the reference
  first_step_mae: mae_bins on the first of the 8 timesteps only (what the robot executes
                  with action_horizon=1)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.contracts import get_contract  # noqa: E402


def load_results(results_dir):
    res = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            d = json.loads(p.read_text())
            d["_file"] = p.name
            res.append(d)
        except Exception as e:  # noqa: BLE001
            print("skip", p, e)
    return res


def fidelity(run, ref, contract):
    P, R = run.get("predictions", {}), ref.get("predictions", {})
    uids = sorted(set(P) & set(R))
    if not uids:
        return {}
    pb = np.stack([np.asarray(P[u]["bins"], dtype=np.float32) for u in uids])
    rb = np.stack([np.asarray(R[u]["bins"], dtype=np.float32) for u in uids])
    exact = float(np.mean([P[u]["text"].strip() == R[u]["text"].strip() for u in uids]))
    d = np.abs(pb - rb)
    out = dict(n=len(uids), exact_match=exact, mae_bins=float(d.mean()),
               mae_bins_per_dim=d.mean(axis=(0, 1)).round(2).tolist(),
               first_step_mae_bins=float(d[:, 0].mean()), max_abs_bins=float(d.max()),
               within_10=float((d <= 10).mean()), within_25=float((d <= 25).mean()))
    if contract is not None:
        pa, ra = contract.denormalize(pb), contract.denormalize(rb)
        out["mae_action"] = float(np.abs(pa - ra)[:, :, :6].mean())
        out["gripper_agree"] = float(np.mean(np.sign(pa[:, :, 6]) == np.sign(ra[:, :, 6])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results/summary.json"))
    ap.add_argument("--md", default=str(ROOT / "results/summary.md"))
    args = ap.parse_args()
    runs = [r for r in load_results(args.results) if "engine" in r]
    stats = {}
    for name in ("vla0", "vla0-smol"):
        try:
            stats[name] = get_contract(name)
        except Exception as e:  # noqa: BLE001
            print("no contract stats for", name, e); stats[name] = None

    rows = []
    for r in runs:
        same_model = [x for x in runs if x["model"] == r["model"] and x.get("subset") == r.get("subset")]
        ref = next((x for x in same_model if x["engine"] == "hf" and x["quant"] == "bf16" and not x.get("tag")), None)
        if ref is None:
            ref = next((x for x in same_model if x["quant"] == "bf16" and not x.get("tag")), None)
        fid = fidelity(r, ref, stats.get(r["model"])) if ref is not None and ref is not r else {}
        # same-engine bf16 reference separates "what quantization changed" from engine-to-engine bf16 noise
        eref = next((x for x in same_model if x["engine"] == r["engine"] and x["quant"] == "bf16" and not x.get("tag")), None)
        efid = fidelity(r, eref, stats.get(r["model"])) if eref is not None and eref is not r and eref is not ref else {}
        lm = r.get("latency_ms", {})
        thr = r.get("throughput", {})
        mp = Path(r.get("model_path", ""))
        weights_gb = None
        if mp.is_dir():
            weights_gb = round(sum(f.stat().st_size for f in mp.glob("*.safetensors")) / 1e9, 2) or None
        if r["engine"].startswith("executorch"):
            weights_gb = round(((r.get("env") or {}).get("pte_mb", 0) + (r.get("env") or {}).get("ptd_mb", 0)) / 1e3, 2) or None
        rows.append(dict(
            model=r["model"], engine=r["engine"], quant=r["quant"], tag=r.get("tag", ""), subset=r.get("subset"),
            n=r.get("n_obs"), lat_mean_ms=lm.get("mean"), lat_p50_ms=lm.get("median"), lat_p90_ms=lm.get("p90"),
            ttft_ms=(r.get("ttft_ms") or {}).get("mean"),
            out_tok=(r.get("out_tokens") or {}).get("mean"), dec_tok_s=r.get("decode_tok_per_s_batch1"),
            thr_b8=(thr.get("8") or {}).get("obs_per_s"), thr_b32=(thr.get("32") or {}).get("obs_per_s"),
            tok_s_b32=(thr.get("32") or {}).get("out_tok_per_s"),
            gpu_mib=(r.get("gpu_mem_used_mib_after_load") or 0) - (r.get("gpu_mem_used_mib_before_load") or 0)
            if r.get("gpu_mem_used_mib_after_load") else None,
            well_formed=r.get("well_formed_rate"), weights_gb=weights_gb, ref=(ref["_file"] if ref is not None else None), **{f"fid_{k}": v for k, v in fid.items()},
            **{f"efid_{k}": v for k, v in efid.items()},
            file=r["_file"], notes=r.get("notes", "")))
    Path(args.out).write_text(json.dumps(rows, indent=1))

    def f(x, fmt):
        return "" if x is None else format(x, fmt)
    lines = ["| model | engine | quant | tag | n | weights GB | latency mean ms | p90 ms | TTFT ms | out tok | dec tok/s (b=1) | obs/s b=8 | obs/s b=32 | well-formed | MAE bins vs hf bf16 | within 10 | 1st-step MAE | gripper agree | MAE action | MAE bins vs own bf16 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda z: (z["model"], z["engine"], z["quant"], z["tag"])):
        lines.append("| " + " | ".join([
            x["model"], x["engine"], x["quant"], x["tag"] or "", str(x["n"]), f(x["weights_gb"], ".2f"), f(x["lat_mean_ms"], ".1f"), f(x["lat_p90_ms"], ".1f"),
            f(x["ttft_ms"], ".1f"), f(x["out_tok"], ".0f"), f(x["dec_tok_s"], ".0f"), f(x["thr_b8"], ".2f"), f(x["thr_b32"], ".2f"),
            f(x["well_formed"], ".2f"),
            f(x.get("fid_mae_bins"), ".1f"), f(x.get("fid_within_10"), ".2f"), f(x.get("fid_first_step_mae_bins"), ".1f"),
            f(x.get("fid_gripper_agree"), ".3f"), f(x.get("fid_mae_action"), ".4f"), f(x.get("efid_mae_bins"), ".1f")]) + " |")
    Path(args.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
