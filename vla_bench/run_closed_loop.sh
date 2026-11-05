#!/usr/bin/env bash
# Serve one configuration and run the LIBERO closed-loop evaluation against it.
# Usage: ./run_closed_loop.sh <label> <python> <serve args...> -- <eval args...>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
LABEL="$1"; PY="$2"; shift 2
SERVE_ARGS=(); EVAL_ARGS=()
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; EVAL_ARGS=("$@"); break; fi; SERVE_ARGS+=("$1"); shift; done
PORT=$((8100 + RANDOM % 800))
# own session/process group, so the engine's worker processes (vLLM EngineCore, MPI workers) die with it
setsid $PY engines/serve.py "${SERVE_ARGS[@]}" --port $PORT > "logs/serve_$LABEL.log" 2>&1 &
SPID=$!
stop_server() { kill -TERM -- -$SPID 2>/dev/null; kill $SPID 2>/dev/null; sleep 3; kill -KILL -- -$SPID 2>/dev/null; kill -9 $SPID 2>/dev/null; }
for i in $(seq 1 240); do grep -q "ready on" "logs/serve_$LABEL.log" && break; kill -0 $SPID 2>/dev/null || break; sleep 5; done
if ! grep -q "ready on" "logs/serve_$LABEL.log"; then echo "server failed to start for $LABEL"; tail -20 "logs/serve_$LABEL.log"; stop_server; exit 1; fi
./eval_closed_loop.sh --url "http://127.0.0.1:$PORT" --label "$LABEL" "${EVAL_ARGS[@]}"
rc=$?
stop_server
exit $rc
