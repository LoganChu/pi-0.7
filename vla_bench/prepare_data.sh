#!/usr/bin/env bash
# Render the benchmark observation set with the LIBERO (Python 3.8) venv from ../libero_sim.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="$HERE/../libero_sim"
export LIBERO_CONFIG_PATH="$SIM/.libero"
export PYTHONPATH="${PYTHONPATH:-}:$SIM/openpi/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$SIM/openpi/examples/libero/.venv/bin/python" "$HERE/prepare_data.py" "$@"
