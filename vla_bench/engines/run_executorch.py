"""VLA-0-Smol on the ExecuTorch runtime (XNNPACK on CPU, or the CUDA/AOTInductor backend).

Drives the three exported methods from Python: token_embedding, vision_encoder (once
per camera) and text_decoder (static KV cache, one call per generated token). Greedy,
stops at <end_of_utterance>. Preprocessing (SmolVLM processor) is identical to the HF
and vLLM runs. Batch size is 1 by construction (the exported cache has batch 1).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import vla0_smol  # noqa: E402
from common.contracts import get_contract  # noqa: E402
from engines.base import GenOut, add_common_args, gpu_mem_used_mib, load_requests, run_benchmark  # noqa: E402


def build(args, contract):

    import torch
    from transformers import AutoProcessor
    from executorch.runtime import Runtime, Verification
    import executorch

    if args.threads:
        from executorch.extension.pybindings import portable_lib as _pl
        _pl._unsafe_reset_threadpool(args.threads)
        torch.set_num_threads(args.threads)
        print("executorch threadpool:", _pl._threadpool_get_thread_count())
    pte = Path(args.pte_dir) / "model.pte"
    ptds = [p for p in Path(args.pte_dir).glob("*.ptd")]
    mem0 = gpu_mem_used_mib()
    t0 = time.time()
    rt = Runtime.get()
    program = rt.load_program(pte, verification=Verification.Minimal, data_path=str(ptds[0]) if ptds else None)
    m_emb = program.load_method("token_embedding")
    m_enc = program.load_method("vision_encoder")
    m_dec = program.load_method("text_decoder")
    meta = {}
    for name in ["get_max_seq_len", "vision_token_id", "get_eos_id", "get_bos_id", "use_kv_cache"]:
        try:
            meta[name] = program.load_method(name).execute([])[0]
        except Exception:
            pass
    load_s = time.time() - t0
    max_seq_len = int(meta.get("get_max_seq_len", 768))
    image_token_id = int(meta.get("vision_token_id", vla0_smol.IMAGE_TOKEN_ID))
    print("loaded", pte, "methods:", program.method_names, "meta:", meta, f"{load_s:.1f}s")

    processor = AutoProcessor.from_pretrained(args.model, **contract.mm_processor_kwargs)
    tok = processor.tokenizer
    # the exported methods take inputs in the dtype the model was exported with (bf16 for the CUDA recipe)
    ea = Path(args.pte_dir) / "export_args.txt"
    in_dtype = torch.bfloat16 if ea.exists() and "--dtype bfloat16" in ea.read_text() else torch.float32

    def infer(batch):
        outs = []
        for r in batch:
            inputs = processor(text=[r.prompt], images=[r.images], return_tensors="pt")
            ids = inputs["input_ids"]
            pv = inputs["pixel_values"].to(in_dtype)      # (1, n_img, 3, 512, 512)
            t_start = time.perf_counter()
            emb = m_emb.execute([ids])[0].clone()          # (1, L, H)
            feats = [m_enc.execute([pv[:, i].contiguous()])[0] for i in range(pv.shape[1])]
            feats = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)
            mask = ids[0] == image_token_id
            assert int(mask.sum()) == feats.shape[0], (int(mask.sum()), feats.shape)
            emb[0, mask] = feats.to(emb.dtype)
            L = ids.shape[1]
            logits = m_dec.execute([emb, torch.arange(L, dtype=torch.long)])[0]
            t_first = time.perf_counter()
            nxt = int(torch.argmax(logits[0, -1].float()))
            gen = [nxt]
            pos = L
            while nxt != contract.eos_token_id and len(gen) < contract.max_new_tokens and pos < max_seq_len - 1:
                e = m_emb.execute([torch.tensor([[nxt]], dtype=torch.long)])[0]
                logits = m_dec.execute([e, torch.tensor([pos], dtype=torch.long)])[0]
                nxt = int(torch.argmax(logits[0, -1].float()))
                gen.append(nxt)
                pos += 1
            if gen and gen[-1] == contract.eos_token_id:
                gen = gen[:-1]
            text = tok.decode(gen, skip_special_tokens=True)
            outs.append(GenOut(text=text, n_prompt_tokens=L, n_out_tokens=len(gen),
                               ttft_ms=(t_first - t_start) * 1000,
                               extra=dict(decode_ms_per_tok=((time.perf_counter() - t_first) * 1000 / max(len(gen), 1)))))
        return outs

    env = dict(executorch=getattr(executorch, "__version__", "1.5.1"), torch=torch.__version__, backend=args.backend,
               pte_dir=str(args.pte_dir), export_args=(Path(args.pte_dir) / "export_args.txt").read_text()
               if (Path(args.pte_dir) / "export_args.txt").exists() else "", load_s=load_s, threads=args.threads or os.cpu_count(),
               pte_mb=round(pte.stat().st_size / 1e6, 1), ptd_mb=round(sum(p.stat().st_size for p in ptds) / 1e6, 1))
    return infer, env, mem0, f"executorch-{args.backend}"


def parse_extra(ap, argv):
    ap.add_argument("--pte-dir", required=True, help="directory with model.pte (+ .ptd)")
    ap.add_argument("--backend", default="xnnpack", help="label only: xnnpack | cuda")
    ap.add_argument("--threads", type=int, default=None, help="XNNPACK threadpool size")
    return ap.parse_args(argv)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--pte-dir", required=True, help="directory with model.pte (+ .ptd)")
    ap.add_argument("--backend", default="xnnpack", help="label only: xnnpack | cuda")
    ap.add_argument("--threads", type=int, default=None, help="XNNPACK threadpool size")
    args = ap.parse_args()
    contract = get_contract(args.model_name, args.model)
    args.model = str(contract.model_dir)
    infer, env, mem0, engine = build(args, contract)
    reqs = load_requests(args, contract)
    run_benchmark(args, engine, infer, reqs, env, contract, mem_before_load=mem0, max_batch=1)


if __name__ == "__main__":
    main()
