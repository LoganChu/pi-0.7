#!/usr/bin/env bash
# Serve pi0.5 (fine-tuned on LIBERO) over a websocket on port 8000.
# First run downloads the ~11.6 GB checkpoint to ~/.cache/openpi.
# Pass extra serve_policy.py args through, e.g. to use pi0-FAST instead:
#   ./serve.sh policy:checkpoint --policy.config pi0_fast_libero --policy.dir gs://openpi-assets/checkpoints/pi0_fast_libero
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/openpi"
exec uv run scripts/serve_policy.py --env LIBERO "$@"
