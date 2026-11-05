"""Pre-compile FlashInfer's sm120 GEMM JIT modules in a standalone process.

vLLM compiles these lazily inside the engine process on first use (29 nvcc units for the
FP8 GEMM alone); together with the ~10 GB the engine already holds that exceeded the
per-job memory cap. Built here once, they are picked up from ~/.cache/flashinfer.
"""
import time
from flashinfer.jit.gemm.core import gen_gemm_sm120_module, gen_gemm_sm120_module_cutlass_fp4

for name, gen in [("gemm_sm120", gen_gemm_sm120_module), ("gemm_sm120_cutlass_fp4", gen_gemm_sm120_module_cutlass_fp4)]:
    t0 = time.time()
    try:
        spec = gen()
        spec.build_and_load()
        print(f"{name}: built/loaded in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAILED {e!r}"[:400], flush=True)
