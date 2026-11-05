# Configurations that could not run on this machine (RTX 5090, SM120) and why

| model | engine | quant | outcome |
|---|---|---|---|
| vla0, vla0-smol | vLLM 0.33.0 | int8 (W8A8, llm-compressor GPTQ) | checkpoint builds, engine refuses to start: `dispatch_scaled_mm ... Int8 not supported on SM120. Use FP8 quantization instead, or run on older arch (SM < 100)` |
| vla0 | TensorRT-LLM 1.2.1 | int4 (W4A16 AWQ, ModelOpt) | checkpoint builds, executor fails: `Not Implemented: SM120 GEMM only supports nvfp4. (cutlass_heuristic.cpp:566)` |
| vla0-smol | PyTorch + torchao 0.18 eager | int8 (W8A8 dynamic) | ran at 220-270 s per chunk with degenerate action strings; stopped by hand after 6 observations (see notes_vla0-smol__hf__int8.txt) |
| vla0-smol | llm-compressor 0.15 | int4 via AWQ | `AWQ needs to match a single smoothlayer for each mapping`: default mappings do not resolve on SmolVLM's module tree; GPTQ W4A16 (group 32) used instead |
| vla0 | ExecuTorch 1.5.1 | any | Qwen2.5-VL is not exportable: optimum-executorch's multimodal path needs a single non-text modality and a fixed-shape vision encoder; Qwen2.5-VL has dynamic-resolution windowed attention and M-RoPE |
| vla0-smol | TensorRT-LLM 1.2.1 | any | SmolVLM / Idefics3 is not in TensorRT-LLM's supported multimodal architectures |
| vla0-smol | ExecuTorch 1.5.1 (prebuilt wheel) | custom-SDPA / custom-KV-cache exports | export succeeds, but neither the portable Python bindings nor the bundled C++ `MultimodalRunner` register `llama::custom_sdpa.out` / `llama::update_cache.out` ("96 instructions don't have corresponding operator registered"); needs a source-built runtime. The standard static-cache export is benchmarked instead |
