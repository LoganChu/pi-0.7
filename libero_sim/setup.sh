#!/usr/bin/env bash
# One-time setup: clone openpi, build the policy-server env (Python 3.11, JAX)
# and the simulator-client env (Python 3.8, LIBERO/robosuite/MuJoCo).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="$HERE/openpi"
OPENPI_REV="215abfb"  # tested commit (2026-08-24)

if [ ! -d "$OPENPI" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
  git -C "$OPENPI" checkout "$OPENPI_REV"
  git -C "$OPENPI" submodule update --init --recursive
fi

echo "== policy server env (openpi, Python 3.11) =="
(cd "$OPENPI" && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .)

echo "== simulator client env (LIBERO, Python 3.8) =="
VENV="$OPENPI/examples/libero/.venv"
(
  cd "$OPENPI"
  [ -d "$VENV" ] || uv venv --python 3.8 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
  VIRTUAL_ENV="$VENV" uv pip install -e packages/openpi-client -e third_party/libero
)

# LIBERO asks interactive questions on first import unless a config exists.
LIBERO_ROOT="$OPENPI/third_party/libero/libero/libero"
mkdir -p "$HERE/.libero"
cat > "$HERE/.libero/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $LIBERO_ROOT/../datasets
assets: $LIBERO_ROOT/assets
EOF

echo "Done. Start the server with ./serve.sh, then run ./run_demo.sh"
