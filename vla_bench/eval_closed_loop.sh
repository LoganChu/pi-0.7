#!/usr/bin/env bash
# Closed-loop LIBERO evaluation against a running engines/serve.py, using the LIBERO Python 3.8 venv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="$HERE/../libero_sim"
export LIBERO_CONFIG_PATH="$SIM/.libero"
export PYTHONPATH="${PYTHONPATH:-}:$SIM/openpi/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$SIM/openpi/examples/libero/.venv/bin/python" "$HERE/eval_closed_loop.py" "$@"
