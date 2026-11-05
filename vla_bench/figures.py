"""Figures from results/summary.json (run report.py first). Writes results/figures/*.png.

fig_latency.png  : batch-1 latency per quantization level, one panel per model, one hue per engine
fig_fidelity.png : action error vs latency (speed/accuracy trade-off), one panel per model
fig_closed_loop.png : LIBERO-Goal success per served configuration
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "results/summary.json").read_text())
rows = [r for r in rows if not r.get("tag")]  # core runs only
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical palette (dataviz skill reference, slots 1-4), fixed order by engine
ENGINES = [("hf", "PyTorch / transformers", "#2a78d6"), ("vllm", "vLLM", "#eb6834"),
           ("trtllm", "TensorRT-LLM", "#1baf7a"), ("executorch", "ExecuTorch", "#eda100")]
QUANT_ORDER = ["fp32", "bf16", "fp8", "fp8-kv", "fp8-static", "int8", "int8wo", "int4", "nvfp4"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MODELS = [("vla0", "VLA-0 (Qwen2.5-VL-3B)"), ("vla0-smol", "VLA-0-Smol (SmolVLM2-500M)")]


def engine_key(e):
    return e.split("-")[0] if e.startswith(("executorch", "hf")) else e


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_latency():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor="#fcfcfb")
    for ax, (model, title) in zip(axes, MODELS):
        sel = [r for r in rows if r["model"] == model and r.get("lat_mean_ms")]
        labels, vals, cols, texts = [], [], [], []
        for ek, ename, col in ENGINES:
            for r in sorted((x for x in sel if engine_key(x["engine"]) == ek),
                            key=lambda x: (x["engine"], QUANT_ORDER.index(x["quant"]) if x["quant"] in QUANT_ORDER else 99)):
                sub = r["engine"].replace("executorch-", "").replace("hf-compile", "compile")
                lab = f"{r['quant']}" + (f" ({sub})" if r["engine"] not in ("hf", "vllm", "trtllm") else "")
                labels.append(lab); vals.append(r["lat_mean_ms"]); cols.append(col)
                texts.append(f"{r['lat_mean_ms']:,.0f} ms")
        y = range(len(vals))
        bars = ax.barh(list(y), vals, color=cols, height=0.62)
        for b, t in zip(bars, texts):
            ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2, t, va="center", ha="left", fontsize=8.5, color=TEXT)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9, color=TEXT); ax.invert_yaxis()
        ax.set_xscale("log"); ax.set_xlabel("latency per action chunk, batch 1 (ms, log scale)", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold")
        ax.set_facecolor("#fcfcfb"); style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ENGINES]
    fig.legend(handles, [n for _, n, _ in ENGINES], loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("VLA inference latency by engine and quantization (RTX 5090; ExecuTorch-xnnpack on 8 CPU cores)",
                 x=0.02, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUT / "fig_latency.png", dpi=160)


# hand-placed label offsets (points) for marks that would otherwise collide: (model, engine, quant) -> (dx, dy, ha)
LABEL_POS = {
    ("vla0", "trtllm", "nvfp4"): (-7, -3, "right"), ("vla0", "vllm", "nvfp4"): (7, 4, "left"),
    ("vla0", "trtllm", "fp8"): (-7, 3, "right"), ("vla0", "vllm", "fp8"): (7, 3, "left"),
    ("vla0", "vllm", "fp8-static"): (-7, -4, "right"), ("vla0", "vllm", "int8wo"): (-7, -9, "right"),
    ("vla0", "vllm", "bf16"): (7, -10, "left"), ("vla0", "trtllm", "bf16"): (7, 4, "left"),
    ("vla0-smol", "vllm", "fp8-kv"): (-7, -3, "right"), ("vla0-smol", "vllm", "fp8"): (7, -10, "left"),
    ("vla0-smol", "vllm", "fp8-static"): (7, 5, "left"), ("vla0-smol", "vllm", "bf16"): (-7, -3, "right"),
    ("vla0-smol", "executorch-cuda", "int4"): (-7, -3, "right"),
}


def point_label(r):
    if r["engine"].startswith("executorch"):
        return f"{r['quant']} ({'gpu' if 'cuda' in r['engine'] else 'cpu'})"
    return r["quant"]


def fig_fidelity():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor="#fcfcfb")
    for ax, (model, title) in zip(axes, MODELS):
        sel = [r for r in rows if r["model"] == model and r.get("lat_mean_ms") and r.get("fid_mae_bins") is not None]
        for ek, ename, col in ENGINES:
            pts = [r for r in sel if engine_key(r["engine"]) == ek]
            if not pts:
                continue
            ax.scatter([r["lat_mean_ms"] for r in pts], [r["fid_mae_bins"] for r in pts], s=70, color=col,
                       edgecolor="#fcfcfb", linewidth=2, zorder=3, label=ename)
            for r in pts:
                dx, dy, ha = LABEL_POS.get((model, r["engine"], r["quant"]), (7, 3, "left"))
                ax.annotate(point_label(r), (r["lat_mean_ms"], r["fid_mae_bins"]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=8, color=TEXT, ha=ha)
        ax.set_xscale("log"); ax.set_xlabel("latency per action chunk, batch 1 (ms, log)", color=MUTED, fontsize=9)
        ax.set_ylabel("mean |Δ action bin| vs PyTorch bf16", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold")
        ax.set_facecolor("#fcfcfb"); style(ax); ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right" if model == "vla0" else "upper center")
    fig.suptitle("Speed vs. action fidelity (lower-left is better)", x=0.02, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_fidelity.png", dpi=160)


def fig_closed_loop():
    cl = []
    for p in sorted((ROOT / "results/closed_loop").glob("*.json")):
        d = json.loads(p.read_text())
        eps = d.get("episodes") or []
        if len(eps) < 5:
            continue
        h = d["health"]
        cl.append(dict(model=h["model"], engine=h["engine"], quant=h["quant"], n=len(eps), k=sum(e["success"] for e in eps),
                       lat=sum(e["policy_latency_ms"] for e in eps) / len(eps)))
    if not cl:
        return
    order = {"vla0": 0, "vla0-smol": 1}
    cl.sort(key=lambda c: (order[c["model"]], -c["k"] / c["n"], c["lat"]))
    names = dict(MODELS)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(cl) + 1.8), facecolor="#fcfcfb")
    cols = {ek: col for ek, _, col in ENGINES}
    y = list(range(len(cl)))
    bars = ax.barh(y, [100 * c["k"] / c["n"] for c in cl], color=[cols[engine_key(c["engine"])] for c in cl], height=0.6)
    for b, c in zip(bars, cl):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{c['k']}/{c['n']}   ·   {c['lat']:,.0f} ms per query",
                va="center", ha="left", fontsize=9, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'VLA-0' if c['model'] == 'vla0' else 'VLA-0-Smol'} · {c['quant']}" for c in cl], fontsize=9, color=TEXT)
    ax.invert_yaxis(); ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("LIBERO-Goal success, % of 10 episodes", color=MUTED, fontsize=9)
    ax.set_facecolor("#fcfcfb"); style(ax)
    used = [e for e in ENGINES if any(engine_key(c["engine"]) == e[0] for c in cl)]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in used], [n for _, n, _ in used], frameon=False, fontsize=9,
              labelcolor=TEXT, loc="lower right")
    fig.suptitle("Closed-loop task success by quantization level", x=0.02, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_closed_loop.png", dpi=160)


if __name__ == "__main__":
    fig_latency(); fig_fidelity(); fig_closed_loop()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
