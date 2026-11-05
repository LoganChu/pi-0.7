#!/usr/bin/env bash
# Run a TensorRT-LLM LLM-API script. MPI_Comm_spawn hangs with the pip Open MPI 5 wheel, so the
# worker is pre-launched with TensorRT-LLM's own trtllm-llmapi-launch under mpirun instead.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/.venvs/trtllm/lib:${LD_LIBRARY_PATH:-}"
export OPAL_PREFIX="$HERE/.venvs/trtllm"
export PATH="$HERE/.venvs/trtllm/bin:$PATH"
exec mpirun -n 1 --oversubscribe trtllm-llmapi-launch python "$@"
