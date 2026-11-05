"""Export VLA-0-Smol (SmolVLM2-500M) to ExecuTorch with optimum-executorch.

Produces model.pte (+ a .ptd weight blob for the CUDA backend) holding three methods:
token_embedding, vision_encoder (one 512x512 image -> 64 x 960 embeddings) and
text_decoder (inputs_embeds + cache_position -> logits, static KV cache).

Two things are patched relative to `optimum-cli export executorch`:
  * the task loader uses AutoModelForPreTraining, which has no SmolVLM mapping in
    transformers 5; we load with AutoModelForImageTextToText instead.
  * SmolVLM's get_image_features filters all-zero padding images with data-dependent
    indexing, which torch.export cannot trace; the vision encoder is exported as
    vision_model -> connector on a fixed (1, 3, 512, 512) input.

Quantization (torchao, applied before export):
  --qlinear 8da4w   int8 dynamic activations, int4 weights (XNNPACK's int4 path)
  --qlinear 8da8w   int8 dynamic activations, int8 weights
  --qlinear 8w      int8 weight-only
  --qlinear 4w --qlinear_packing_format tile_packed_to_4d   int4 weight-only (CUDA backend)
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/vla0-smol-libero-hf"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="xnnpack", choices=["xnnpack", "cuda", "portable"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--qlinear", default=None)
    ap.add_argument("--qlinear-group-size", type=int, default=None)
    ap.add_argument("--qlinear-packing-format", default=None)
    ap.add_argument("--qlinear-encoder", default=None)
    ap.add_argument("--qembedding", default=None)
    ap.add_argument("--use-custom-sdpa", action="store_true")
    ap.add_argument("--use-custom-kv-cache", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    import torchao
    from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig
    from optimum.exporters.executorch import integrations
    from optimum.exporters.executorch.convert import export_to_executorch
    from optimum.exporters.executorch.quantization import quantize_model_

    # --- a vision encoder wrapper without SmolVLM's data-dependent padding filter
    class SmolVisionExportable(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def prepare_export_inputs(self):
            vc = self.model.config.vision_config
            return torch.zeros((1, 3, vc.image_size, vc.image_size), dtype=self.model.dtype, device=self.model.device), None

        def forward(self, input_features):
            # Static full-image path of SmolVLMVisionTransformer: the stock forward derives the
            # position ids from a data-dependent patch mask (unbacked SymInts in torch.export);
            # for a full 512x512 image they are simply arange(1024).
            vm = self.model.model.vision_model
            emb = vm.embeddings.patch_embedding(input_features).flatten(2).transpose(1, 2)
            emb = emb + vm.embeddings.position_embedding.weight.unsqueeze(0)
            h = vm.encoder(inputs_embeds=emb, attention_mask=None).last_hidden_state
            h = vm.post_layernorm(h)
            return self.model.model.connector(h)

    integrations.VisionExportableModule = SmolVisionExportable

    t0 = time.time()
    config = AutoConfig.from_pretrained(args.model)
    config.use_cache = True
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, device_map=args.device, dtype=args.dtype, config=config,
        attn_implementation="custom_sdpa" if args.use_custom_sdpa else "sdpa").eval()
    model.generation_config = GenerationConfig(
        use_cache=True, cache_implementation="static", max_length=args.max_seq_len,
        cache_config={"batch_size": 1, "max_cache_len": args.max_seq_len, "device": args.device})
    model.get_decoder().generation_config = model.generation_config
    for p in model.parameters():
        p.requires_grad = False

    def q(target, cfg, group_size=None, packing=None):
        if not cfg:
            return
        kw = {"eager_model": target, "qlinear_config": cfg}
        if group_size:
            kw["qlinear_group_size"] = group_size
        if packing:
            kw["qlinear_packing_format"] = packing
        quantize_model_(**kw)

    q(model.get_decoder(), args.qlinear, args.qlinear_group_size, args.qlinear_packing_format)
    if args.qlinear and not hasattr(model.get_decoder(), "lm_head"):
        q(model.lm_head, args.qlinear, args.qlinear_group_size, args.qlinear_packing_format)
    if args.qlinear_encoder:
        q(model.model.vision_model, args.qlinear_encoder, None, args.qlinear_packing_format)
        q(model.model.connector, args.qlinear_encoder, None, args.qlinear_packing_format)
    if args.qembedding:
        quantize_model_(eager_model=model.get_input_embeddings(), qembedding_config=args.qembedding) if False else None
        from torchao.quantization import IntxWeightOnlyConfig, PerGroup, quantize_
        from torchao.quantization.granularity import PerAxis
        emb_cfg = IntxWeightOnlyConfig(weight_dtype=torch.int8 if args.qembedding == "8w" else torch.int4, granularity=PerAxis(0))
        quantize_(model, emb_cfg, lambda m, fqn: isinstance(m, torch.nn.Embedding))

    exportable = integrations.MultiModalTextToTextExportableModule(
        model=model, modality="vision", encoder_model=model.model.vision_model, max_seq_len=args.max_seq_len,
        processor_config=None, use_custom_kv_cache=args.use_custom_kv_cache, use_custom_sdpa=args.use_custom_sdpa)
    os.makedirs(args.out, exist_ok=True)
    progs = export_to_executorch(model=exportable, task="image-text-to-text", recipe=args.recipe, output_dir=args.out)
    print("exported", list(progs.keys()), "->", args.out, f"in {time.time()-t0:.0f}s")
    for f in sorted(Path(args.out).iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size/1e6:8.1f} MB")
    Path(args.out, "export_args.txt").write_text(" ".join(sys.argv[1:]) + f"\ntorchao={torchao.__version__}\n")


if __name__ == "__main__":
    main()
