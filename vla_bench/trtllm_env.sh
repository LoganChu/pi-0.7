#!/usr/bin/env bash
# Run a command inside the TensorRT-LLM venv with the pip-installed Open MPI visible
# (TensorRT-LLM's C++ runtime links libmpi.so.40; there is no system MPI on this machine).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/.venvs/trtllm/lib:${LD_LIBRARY_PATH:-}"
export OPAL_PREFIX="$HERE/.venvs/trtllm"
export PATH="$HERE/.venvs/trtllm/bin:$PATH"
exec "$HERE/.venvs/trtllm/bin/python" "$@"
