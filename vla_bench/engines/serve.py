"""Minimal HTTP policy server so the LIBERO simulator (Python 3.8 venv) can drive any engine.

POST /act  {"instruction": str, "agentview_png_b64": str, "wrist_png_b64": str, "state": [8 floats]}
        -> {"text": str, "bins": [[...]*7]*8, "action": [[...]*7]*8, "latency_ms": float}
GET  /health -> {"engine": ..., "quant": ...}

Usage (same flags as the engine's run_*.py plus --port):
  .venvs/vllm/bin/python engines/serve.py --engine vllm --model-name vla0 --quant fp8 --port 8008
"""
import argparse
import base64
import importlib
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import vla0, vla0_smol  # noqa: E402
from common.contracts import get_contract  # noqa: E402
from engines.base import Request, add_common_args  # noqa: E402


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--engine", dest="engine_mod", required=True, choices=["vllm", "hf", "trtllm", "executorch"])
    pre.add_argument("--port", type=int, default=8008)
    pre_args, rest = pre.parse_known_args()
    mod = importlib.import_module(f"engines.run_{pre_args.engine_mod}")
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = mod.parse_extra(ap, rest)
    args.engine = None
    args.port = pre_args.port
    contract = get_contract(args.model_name, args.model)
    args.model = str(contract.model_dir)
    infer, env, mem0, engine = mod.build(args, contract)
    print(f"[serve] {engine} / {args.model_name} / {args.quant} ready on :{args.port}", flush=True)

    def decode_img(b64):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._json(200, dict(engine=engine, model=args.model_name, quant=args.quant, env={k: str(v) for k, v in env.items()}))

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            a, w = decode_img(req["agentview_png_b64"]), decode_img(req["wrist_png_b64"])
            if contract.name == "vla0":
                prompt, images = vla0.build_prompt(req["instruction"]), [vla0.tile_obs(a, w)]
            else:
                prompt = vla0_smol.build_prompt(req["instruction"], np.asarray(req["state"], dtype=np.float32), contract.stats)
                images = [vla0_smol.center_crop(a), vla0_smol.center_crop(w)]
            t0 = time.perf_counter()
            out = infer([Request("live", req["instruction"], prompt, images)])[0]
            dt = (time.perf_counter() - t0) * 1000
            bins = contract.parse(out.text)
            action = contract.denormalize(bins)
            self._json(200, dict(text=out.text, bins=bins.tolist(), action=action.tolist(), latency_ms=dt,
                                 n_out_tokens=out.n_out_tokens, well_formed=contract.is_well_formed(out.text)))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    try:
        srv.serve_forever()
    finally:
        if hasattr(mod, "shutdown"):
            mod.shutdown()


if __name__ == "__main__":
    main()
