#!/usr/bin/env bash
# Run the MuJoCo/LIBERO client against a running ./serve.sh. Args go to run_demo.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="$HERE/openpi"
export LIBERO_CONFIG_PATH="$HERE/.libero"
export PYTHONPATH="${PYTHONPATH:-}:$OPENPI/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$OPENPI/examples/libero/.venv/bin/python" "$HERE/run_demo.py" "$@"
