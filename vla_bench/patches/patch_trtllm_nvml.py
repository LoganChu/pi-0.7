"""TensorRT-LLM 1.2.1 x nvidia-ml-py 13.615: `confidential_compute_enabled()` passes a ctypes
struct by value to nvmlSystemGetConfComputeSettings, which segfaults the executor process
(seen as "Segfault encountered ... nvmlSystemGetConfComputeSettings" in the leader node).
Pass it by reference. Idempotent; run once after installing the trtllm venv."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / ".venvs/trtllm/lib/python3.12/site-packages/tensorrt_llm/_utils.py"
s = p.read_text()
old = "pynvml.nvmlSystemGetConfComputeSettings(cc_settings)"
new = "pynvml.nvmlSystemGetConfComputeSettings(__import__('ctypes').byref(cc_settings))"
if new in s:
    print("already patched")
elif old in s:
    p.write_text(s.replace(old, new)); print("patched", p)
else:
    raise SystemExit("pattern not found; TensorRT-LLM version changed?")
