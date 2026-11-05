"""VLA-0 / VLA-0-Smol on vLLM (offline LLM API, v1 engine).

Quant levels:
  bf16        : reference (fp32 checkpoints are cast to bf16)
  fp8         : vLLM on-the-fly FP8 (W8A8, dynamic per-token activation scales)
  fp8-kv      : bf16 weights + FP8 KV cache, to isolate the KV-cache effect
  <anything>  : a pre-quantized checkpoint passed with --model (llm-compressor / ModelOpt /
                AWQ / GPTQ); the label is whatever --quant says, e.g. fp8-static, int8, int4, nvfp4
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
    import vllm
    from vllm import LLM, SamplingParams

    mem0 = gpu_mem_used_mib()
    kwargs = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
                  limit_mm_per_prompt={"image": contract.n_images}, gpu_memory_utilization=args.gpu_mem,
                  enforce_eager=args.enforce_eager, max_num_seqs=64, seed=0,
                  mm_processor_kwargs=contract.mm_processor_kwargs or None)
    q = args.quant
    if q == "fp8":
        kwargs["quantization"] = "fp8"
    elif q == "fp8-kv":
        kwargs["kv_cache_dtype"] = "fp8"
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.kv_dtype:
        kwargs["kv_cache_dtype"] = args.kv_dtype
    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=contract.max_new_tokens, stop_token_ids=[contract.eos_token_id],
                        allowed_token_ids=contract.allowed_token_ids if args.restrict_tokens else None, seed=0)

    def infer(batch):
        inputs = [{"prompt": r.prompt, "multi_modal_data": {"image": r.images if len(r.images) > 1 else r.images[0]}}
                  for r in batch]
        outs = llm.generate(inputs, sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            m = o.metrics
            ttft = None
            if m is not None and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                ttft = (m.first_token_time - m.arrival_time) * 1000
            res.append(GenOut(text=c.text, n_prompt_tokens=len(o.prompt_token_ids or []),
                              n_out_tokens=len(c.token_ids), ttft_ms=ttft))
        return res

    env = dict(vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), load_s=load_s,
               engine_kwargs={k: str(v) for k, v in kwargs.items() if k != "model"}, restrict_tokens=args.restrict_tokens)
    return infer, env, mem0, "vllm"


def parse_extra(ap, argv):
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--restrict-tokens", action="store_true",
                    help="apply VLA-0's digits/space/EOS logits mask via allowed_token_ids")
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--quantization", default=None, help="explicit vLLM quantization backend name")
    return ap.parse_args(argv)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--restrict-tokens", action="store_true",
                    help="apply VLA-0's digits/space/EOS logits mask via allowed_token_ids")
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--quantization", default=None, help="explicit vLLM quantization backend name")
    args = ap.parse_args()
    contract = get_contract(args.model_name, args.model)
    args.model = str(contract.model_dir)
    infer, env, mem0, engine = build(args, contract)
    reqs = load_requests(args, contract)
    run_benchmark(args, engine, infer, reqs, env, contract, mem_before_load=mem0)


if __name__ == "__main__":
    main()
