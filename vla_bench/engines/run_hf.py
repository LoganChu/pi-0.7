"""Plain PyTorch / Hugging Face transformers: the reference implementation (VLA-0's and
VLA-0-Smol's own eval loops call `generate` on exactly these classes) and the baseline
every engine is compared against.

Quant levels (torchao, language-model nn.Linear layers only, vision tower stays bf16):
  bf16   : reference, eager
  fp8    : Float8DynamicActivationFloat8WeightConfig (W8A8 FP8, per-row)
  int8   : Int8DynamicActivationInt8WeightConfig (W8A8 INT8)
  int8wo : Int8WeightOnlyConfig
  int4   : Int4WeightOnlyConfig (group 128)
--compile adds torch.compile on the text decoder forward.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contracts import get_contract  # noqa: E402
from engines.base import GenOut, add_common_args, gpu_mem_used_mib, load_requests, run_benchmark  # noqa: E402


def build(args, contract):

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, args.dtype)
    mem0 = gpu_mem_used_mib()
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation=args.attn).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, **contract.mm_processor_kwargs)
    processor.tokenizer.padding_side = "left"

    q = args.quant
    if q not in ("bf16", "fp32", "fp16"):
        import torchao
        from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, Int4WeightOnlyConfig,
                                          Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, PerRow, quantize_)
        cfgs = {"fp8": lambda: Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
                "int8": lambda: Int8DynamicActivationInt8WeightConfig(),
                "int8wo": lambda: Int8WeightOnlyConfig(),
                # tile_packed_to_4d = the tinygemm bf16 kernel; torchao 0.18's default int4 packing needs the mslk package
                "int4": lambda: Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d")}
        if q not in cfgs:
            raise SystemExit(f"unknown quant {q}; choose from {list(cfgs)}")
        lm = model.get_decoder() if hasattr(model, "get_decoder") else model.model
        quantize_(lm, cfgs[q]())
        quantize_(model.lm_head, cfgs[q]())
        print("quantized with torchao", torchao.__version__, q)
    load_s = time.time() - t0

    if args.compile:
        dec = model.get_decoder()
        dec.forward = torch.compile(dec.forward, mode="reduce-overhead", dynamic=True)

    logits_processor = None
    if args.restrict_tokens and contract.allowed_token_ids:
        from transformers import LogitsProcessor, LogitsProcessorList

        class Allowed(LogitsProcessor):
            def __init__(self, ids):
                self.mask = None; self.ids = ids
            def __call__(self, input_ids, scores):
                if self.mask is None or self.mask.shape[-1] != scores.shape[-1]:
                    self.mask = torch.full((scores.shape[-1],), float("-inf"), device=scores.device)
                    self.mask[self.ids] = 0
                return scores + self.mask
        logits_processor = LogitsProcessorList([Allowed(contract.allowed_token_ids)])

    @torch.inference_mode()
    def infer(batch):
        texts = [r.prompt for r in batch]
        images = [r.images if contract.n_images > 1 else r.images[0] for r in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(args.device)
        if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=contract.max_new_tokens,
                             eos_token_id=contract.eos_token_id, pad_token_id=contract.pad_token_id,
                             logits_processor=logits_processor)
        n_in = inputs["input_ids"].shape[1]
        outs = []
        for i in range(len(batch)):
            ids = gen[i, n_in:].tolist()
            ids = ids[: ids.index(contract.eos_token_id)] if contract.eos_token_id in ids else ids
            ids = [t for t in ids if t != contract.pad_token_id]
            outs.append(GenOut(text=processor.tokenizer.decode(ids, skip_special_tokens=True),
                               n_prompt_tokens=int(inputs["attention_mask"][i].sum()), n_out_tokens=len(ids)))
        return outs

    env = dict(transformers=transformers.__version__, torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), attn=args.attn, compile=args.compile, load_s=load_s, dtype=args.dtype)
    return infer, env, mem0, "hf" + ("-compile" if args.compile else "")


def parse_extra(ap, argv):
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--restrict-tokens", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    return ap.parse_args(argv)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--restrict-tokens", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    contract = get_contract(args.model_name, args.model)
    args.model = str(contract.model_dir)
    infer, env, mem0, engine = build(args, contract)
    reqs = load_requests(args, contract)
    run_benchmark(args, engine, infer, reqs, env, contract, mem_before_load=mem0, max_batch=32)
    import torch
    print("peak torch alloc MiB:", torch.cuda.max_memory_allocated() // 2**20)


if __name__ == "__main__":
    main()
