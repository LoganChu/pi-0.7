"""Post-training quantization of VLA-0 (Qwen2.5-VL-3B) with NVIDIA ModelOpt, exported in the
Hugging Face layout (+ hf_quant_config.json) that TensorRT-LLM's PyTorch backend loads directly.

  --qformat fp8       W8A8 FP8, per-tensor static activation scales (calibrated)
  --qformat int4_awq  W4A16 INT4 AWQ, block 128
  --qformat nvfp4     W4A4 NVFP4 (block 16, FP8 block scales), Blackwell
  --kv fp8            also calibrate an FP8 KV cache

Only the language model is quantized; the SigLIP-style vision tower and the merger stay bf16
(ModelOpt's own VLM recipe does the same). Calibration runs the *full* multimodal forward on
the benchmark's LIBERO observations, prompt + the bf16 model's action text, so the activation
ranges cover the image prefill and the digit-only decode regime.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import vla0  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/vla0-libero-hf"))
    ap.add_argument("--qformat", required=True, choices=["fp8", "int4_awq", "nvfp4", "int8_sq"])
    ap.add_argument("--kv", default=None, choices=[None, "fp8"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--ref", default=str(ROOT / "results/vla0__vllm__bf16__all.json"))
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    out = args.out or str(ROOT / f"models/vla0-libero-modelopt-{args.qformat}{'-kv' + args.kv if args.kv else ''}")

    import torch
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=vla0.MIN_PIXELS, max_pixels=vla0.MAX_PIXELS)
    processor.tokenizer.padding_side = "left"

    obs = vla0.load_manifest(args.data, subset="all")[: args.n_calib]
    ref = json.loads(Path(args.ref).read_text())["predictions"] if Path(args.ref).exists() else {}
    samples = []
    for o in obs:
        text = vla0.build_prompt(o.instruction)
        if o.uid in ref:
            text += ref[o.uid]["text"] + vla0.EOS_TOKEN
        samples.append((text, o.tiled(Path(args.data))))
    print(f"calibration: {len(samples)} samples, completions from bf16 run: {bool(ref)}")

    @torch.no_grad()
    def forward_loop(m):
        for i in range(0, len(samples), args.batch):
            chunk = samples[i:i + args.batch]
            enc = processor(text=[t for t, _ in chunk], images=[im for _, im in chunk], return_tensors="pt", padding=True).to("cuda")
            model(**enc)

    cfg = {"fp8": mtq.FP8_DEFAULT_CFG, "int4_awq": mtq.INT4_AWQ_CFG, "nvfp4": mtq.NVFP4_DEFAULT_CFG,
           "int8_sq": mtq.INT8_SMOOTHQUANT_CFG}[args.qformat]
    import copy
    cfg = copy.deepcopy(cfg)
    # keep the vision tower, merger and lm_head in bf16
    cfg["quant_cfg"]["*visual*"] = {"enable": False}
    cfg["quant_cfg"]["*merger*"] = {"enable": False}
    cfg["quant_cfg"]["*lm_head*"] = {"enable": False}
    if args.kv == "fp8":
        cfg["quant_cfg"]["*[kv]_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None, "enable": True}
        cfg["quant_cfg"]["*k_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None, "enable": True}
        cfg["quant_cfg"]["*v_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None, "enable": True}

    lm = model.model.language_model if hasattr(model, "model") and hasattr(model.model, "language_model") else model.language_model
    mtq.quantize(model, cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(lm)

    Path(out).mkdir(parents=True, exist_ok=True)
    export_hf_checkpoint(model, export_dir=out)
    processor.save_pretrained(out)
    for f in ["action_stats.json"]:
        if (Path(args.model) / f).exists():
            (Path(out) / f).write_text((Path(args.model) / f).read_text())
    print("exported ->", out)
    print((Path(out) / "hf_quant_config.json").read_text() if (Path(out) / "hf_quant_config.json").exists() else "no hf_quant_config.json?!")


if __name__ == "__main__":
    main()
