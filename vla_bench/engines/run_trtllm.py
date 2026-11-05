"""VLA-0 on TensorRT-LLM (PyTorch backend, LLM API, in-process).

Quant levels are whatever checkpoint --model points to:
  bf16   : the plain HF checkpoint
  fp8    : ModelOpt FP8 export (W8A8 FP8 per-tensor static scales, FP8 KV cache optional)
  int4   : ModelOpt INT4-AWQ export (W4A16, group 128)
  nvfp4  : ModelOpt NVFP4 export (W4A4 FP4 block-16, Blackwell native)
Decoding is greedy; the prompt is the pre-rendered VLA-0 chat string with one
<|image_pad|> placeholder that TensorRT-LLM's Qwen2.5-VL input processor expands.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contracts import get_contract  # noqa: E402
from engines.base import GenOut, add_common_args, gpu_mem_used_mib, load_requests, run_benchmark  # noqa: E402


_LLM = None


def shutdown():
    if _LLM is not None:
        _LLM.shutdown()


def build(args, contract):

    import torch
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig

    mem0 = gpu_mem_used_mib()
    kv = KvCacheConfig(free_gpu_memory_fraction=args.gpu_mem, enable_block_reuse=False)
    if args.kv_dtype:
        kv.dtype = args.kv_dtype
    kwargs = dict(model=args.model, backend="pytorch", kv_cache_config=kv, max_seq_len=args.max_seq_len,
                  max_batch_size=args.max_batch, max_num_tokens=8192,
                  cuda_graph_config=None if args.no_cuda_graph else CudaGraphConfig(enable_padding=True, max_batch_size=args.max_batch))
    if args.attn_backend:
        kwargs["attn_backend"] = args.attn_backend
    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0
    global _LLM
    _LLM = llm
    sp = SamplingParams(max_tokens=contract.max_new_tokens, temperature=0.0, top_k=1,
                        end_id=contract.eos_token_id, stop_token_ids=[contract.eos_token_id], detokenize=True)

    def infer(batch):
        inputs = [{"prompt": r.prompt, "multi_modal_data": {"image": r.images}} for r in batch]
        outs = llm.generate(inputs, sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            if ids and ids[-1] == contract.eos_token_id:
                ids = ids[:-1]
            res.append(GenOut(text=c.text, n_prompt_tokens=len(o.prompt_token_ids or []), n_out_tokens=len(ids)))
        return res

    env = dict(tensorrt_llm=tensorrt_llm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), load_s=load_s, kv_dtype=args.kv_dtype, attn_backend=args.attn_backend,
               cuda_graph=not args.no_cuda_graph)
    return infer, env, mem0, "trtllm"


def parse_extra(ap, argv):
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--kv-dtype", default=None, help="e.g. fp8")
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--attn-backend", default=None, help="TRTLLM | FLASHINFER | VANILLA")
    ap.add_argument("--max-batch", type=int, default=64)
    return ap.parse_args(argv)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--kv-dtype", default=None, help="e.g. fp8")
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--attn-backend", default=None, help="TRTLLM | FLASHINFER | VANILLA")
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    contract = get_contract(args.model_name, args.model)
    args.model = str(contract.model_dir)
    infer, env, mem0, engine = build(args, contract)
    reqs = load_requests(args, contract)
    try:
        run_benchmark(args, engine, infer, reqs, env, contract, mem_before_load=mem0)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
