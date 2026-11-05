"""Post-training quantization with llm-compressor -> compressed-tensors checkpoints that vLLM
loads natively. Works for both benchmarked VLAs (the contract supplies prompt + images).

Recipes (language-model linears only; vision tower, projector/merger and lm_head stay bf16,
which is also what NVIDIA ModelOpt and the quantized-VLM literature do):
  fp8-static : W8A8 FP8, per-channel weights, static per-tensor activation scales
  int8       : W8A8 INT8 via GPTQ (per-channel weights, dynamic per-token acts) -- vLLM rejects it on SM120
  int8wo     : W8A16 INT8 weight-only via GPTQ (Marlin kernels)
  int4       : W4A16 via AWQ, group size 128, symmetric
  int4-gptq  : W4A16 via GPTQ, group size 128
  nvfp4      : W4A4 NVFP4 (block 16 FP4 with FP8 scales), Blackwell native

Calibration uses the benchmark's own LIBERO observations with each prompt completed by the
bf16 model's action text, so activation statistics cover both the image prefill and the
digit-token decode regime.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import vla0  # noqa: E402
from common.contracts import get_contract  # noqa: E402

IGNORE = {
    "vla0": ["re:.*visual.*", "re:.*lm_head.*", "re:.*merger.*"],
    "vla0-smol": ["re:.*vision_model.*", "re:.*connector.*", "re:.*lm_head.*"],
}
EOS = {"vla0": vla0.EOS_TOKEN, "vla0-smol": "<end_of_utterance>"}


def build_calib(contract, processor, data_dir, ref_json, n):
    from datasets import Dataset
    obs = vla0.load_manifest(data_dir, subset="all")
    ref = json.loads(Path(ref_json).read_text())["predictions"] if ref_json else {}
    rows = []
    for o in obs[:n]:
        text, images = contract.build(o, Path(data_dir))
        if o.uid in ref:
            text += ref[o.uid]["text"] + EOS[contract.name]
        enc = processor(text=[text], images=[images] if contract.n_images > 1 else [images[0]], return_tensors="pt", padding=False)
        # numpy, not nested Python lists: 120 samples of pixel values as Python floats cost ~16 GB of host RAM
        rows.append({k: v.numpy() for k, v in enc.items() if hasattr(v, "numpy")})
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="vla0", choices=["vla0", "vla0-smol"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--recipe", required=True, choices=["fp8-static", "int8", "int8wo", "int4", "int4-gptq", "nvfp4"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--ref", default=None, help="bf16 predictions json (default results/<model>__vllm__bf16__all.json)")
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--group-size", type=int, default=None,
                    help="weight group size for int4/int8wo (default 128; 32 for VLA-0-Smol, whose hidden size 960 is not a multiple of 128)")
    args = ap.parse_args()
    if args.group_size is None:
        args.group_size = 32 if args.model_name == "vla0-smol" else 128
    contract = get_contract(args.model_name, args.model)
    out = args.out or str(ROOT / f"models/{contract.model_dir.name.replace('-hf', '')}-llmc-{args.recipe}")
    print("recipe", args.recipe, "group_size", args.group_size, "->", out)
    ref = args.ref or str(ROOT / f"results/{args.model_name}__vllm__bf16__all.json")
    ref = ref if Path(ref).exists() else None

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.awq import AWQModifier
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    def wq_scheme(bits):
        return {"group_0": QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True, strategy="group", group_size=args.group_size))}

    model = AutoModelForImageTextToText.from_pretrained(str(contract.model_dir), dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(str(contract.model_dir), **contract.mm_processor_kwargs)
    ds = build_calib(contract, processor, args.data, ref, args.n_calib)
    print("calibration samples:", len(ds), "columns:", ds.column_names, "with bf16 completions:", bool(ref))

    ignore = IGNORE[args.model_name]
    r = args.recipe
    if r == "fp8-static":
        recipe = [QuantizationModifier(targets="Linear", scheme="FP8", ignore=ignore)]
    elif r == "int8":
        # SmoothQuant's default mappings do not resolve on these VLM module trees; plain GPTQ W8A8.
        recipe = [GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore)]
    elif r == "int8wo":
        # weight-only INT8 (W8A16): vLLM's INT8 W8A8 cutlass kernels refuse SM120 (consumer Blackwell)
        recipe = [GPTQModifier(config_groups=wq_scheme(8), ignore=ignore)]
    elif r == "int4":
        recipe = [AWQModifier(config_groups=wq_scheme(4), ignore=ignore)]
    elif r == "int4-gptq":
        recipe = [GPTQModifier(config_groups=wq_scheme(4), ignore=ignore)]
    elif r == "nvfp4":
        recipe = [QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)]

    def collate(batch):
        assert len(batch) == 1
        return {k: torch.tensor(v) for k, v in batch[0].items()}

    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=1024, num_calibration_samples=len(ds),
            data_collator=collate, output_dir=out, save_compressed=True)
    processor.save_pretrained(out)
    for f in ["action_stats.json", "preprocessor_config.json"]:
        if (contract.model_dir / f).exists():
            (Path(out) / f).write_text((contract.model_dir / f).read_text())
    print("saved ->", out)


if __name__ == "__main__":
    main()
