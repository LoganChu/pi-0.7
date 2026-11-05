"""Compact per-model tables for the README, generated from results/summary.json and
results/closed_loop/*.json so no number is transcribed by hand. Writes results/tables.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [r for r in json.loads((ROOT / "results/summary.json").read_text()) if not r.get("tag") or r["tag"] == "fp8kv"]
cl = {}
for p in sorted((ROOT / "results/closed_loop").glob("*.json")):
    d = json.loads(p.read_text())
    if d.get("episodes"):
        h = d["health"]
        eps = d["episodes"]
        cl[(h["model"], h["engine"], h["quant"])] = dict(
            sr=sum(e["success"] for e in eps) / len(eps), n=len(eps),
            lat=sum(e["policy_latency_ms"] for e in eps) / len(eps),
            steps=sum(e["steps"] for e in eps) / len(eps), malformed=sum(e["malformed"] for e in eps))

ENGINE_NAME = {"hf": "PyTorch (eager)", "vllm": "vLLM", "trtllm": "TensorRT-LLM",
               "executorch-xnnpack": "ExecuTorch XNNPACK (CPU)", "executorch-cuda": "ExecuTorch CUDA"}
ENGINE_ORDER = list(ENGINE_NAME)
QORDER = ["fp32", "bf16", "fp8", "fp8-static", "fp8-kv", "int8", "int8wo", "int4", "nvfp4"]


def f(x, fmt, dash="–"):
    return dash if x is None else format(x, fmt)


out = []
for model, title, bins in [("vla0", "VLA-0 (Qwen2.5-VL-3B, 3.75 B params, 56 integers in 0–1000)", 1000),
                           ("vla0-smol", "VLA-0-Smol (SmolVLM2-500M, 56 integers in 0–511)", 512)]:
    sel = [r for r in rows if r["model"] == model]
    sel.sort(key=lambda r: (ENGINE_ORDER.index(r["engine"]) if r["engine"] in ENGINE_ORDER else 9,
                            QORDER.index(r["quant"]) if r["quant"] in QORDER else 9, r.get("tag") or ""))
    out += [f"### {title}", "",
            "| engine | level | checkpoint GB | latency ms (b=1) | Hz | tok/s | obs/s b=8 | obs/s b=32 | well-formed | Δ bins vs PyTorch bf16 | Δ bins vs own bf16 | within 10 bins | closed-loop success |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sel:
        q = r["quant"] + (" + FP8 KV" if r.get("tag") == "fp8kv" else "")
        c = cl.get((model, r["engine"], r["quant"])) if not r.get("tag") else None
        lat = r["lat_mean_ms"]
        out.append("| " + " | ".join([
            ENGINE_NAME.get(r["engine"], r["engine"]), q, f(r.get("weights_gb"), ".2f"), f(lat, ",.0f"),
            f(1000 / lat if lat else None, ".2f"), f(r.get("dec_tok_s"), ".0f"), f(r.get("thr_b8"), ".1f"), f(r.get("thr_b32"), ".1f"),
            f(r.get("well_formed"), ".0%"), f(r.get("fid_mae_bins"), ".1f"), f(r.get("efid_mae_bins"), ".1f"),
            f(r.get("fid_within_10"), ".0%"),
            (f"{c['sr']:.0%} ({round(c['sr'] * c['n'])}/{c['n']})" if c else "–")]) + " |")
    out.append("")

if cl:
    out += ["### Closed loop (LIBERO-Goal, 10 tasks × 1 initial state, each paper's own protocol)", "",
            "| model | engine | level | success | mean steps | policy latency ms | malformed outputs |", "|---|---|---|---|---|---|---|"]
    for (model, engine, quant), c in sorted(cl.items()):
        out.append(f"| {model} | {ENGINE_NAME.get(engine, engine)} | {quant} | {c['sr']:.0%} ({round(c['sr'] * c['n'])}/{c['n']}) | "
                   f"{c['steps']:.0f} | {c['lat']:,.0f} | {c['malformed']} |")
    out.append("")
(ROOT / "results/tables.md").write_text("\n".join(out))
print("\n".join(out))
