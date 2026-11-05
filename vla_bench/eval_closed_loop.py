"""Closed-loop LIBERO success rate for a policy served by engines/serve.py.

Runs under the LIBERO Python 3.8 venv (./eval_closed_loop.sh). Reproduces the two
models' published evaluation protocols:
  vla0      : query every step (action_horizon=1), ensemble the last 8 chunks (VLA-0
              eval_libero.py --action_horizon 1 --ensemble_prediction 8, ensemble_version 1)
  vla0-smol : query every step, uniform temporal ensemble of the last 8 chunks
              (lerobot-eval --policy.n_action_steps=0 --policy.ensemble_size=8)
Writes results/closed_loop/<label>.json with per-episode success, steps and policy latency.
"""
import argparse
import base64
import io
import json
import math
import os
import pathlib
import time
import urllib.request

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLE = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def png_b64(img):
    buf = io.BytesIO(); imageio.imwrite(buf, np.ascontiguousarray(img[::-1, ::-1]), format="png")
    return base64.b64encode(buf.getvalue()).decode()


def query(url, obs, instruction):
    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    body = json.dumps(dict(instruction=instruction, agentview_png_b64=png_b64(obs["agentview_image"]),
                           wrist_png_b64=png_b64(obs["robot0_eye_in_hand_image"]), state=state.tolist())).encode()
    req = urllib.request.Request(url + "/act", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    out["rtt_ms"] = (time.perf_counter() - t0) * 1000
    return out


class VLA0Ensemble:
    """VLA-0 eval_run ensemble_version=1 with action_horizon=1."""
    def __init__(self, n=8):
        self.n = n; self.old = []

    def update(self, chunk):
        self.old.append(np.asarray(chunk, dtype=np.float64))
        if len(self.old) > self.n:
            self.old.pop(0)
        acc = np.zeros_like(self.old[-1]); cnt = np.zeros_like(self.old[-1]); new_old = []
        for c in self.old[:-1]:
            if len(c) <= 1:
                continue
            c = c[1:]; new_old.append(c)
            acc[:len(c)] += 0.5 * c; cnt[:len(c)] += 0.5
        new_old.append(self.old[-1]); acc += self.old[-1]; cnt += 1
        self.old = new_old
        return (acc / cnt)[0]


class SmolEnsemble:
    """LeRobot VLA0TemporalEnsembler: mean over the last n chunks at their offsets."""
    def __init__(self, n=8):
        self.n = n; self.sched = []

    def update(self, chunk):
        self.sched.append(np.asarray(chunk, dtype=np.float64))
        if len(self.sched) > self.n:
            self.sched.pop(0)
        acts = []
        for i, s in enumerate(reversed(self.sched)):
            if i < len(s):
                acts.append(s[i])
        return np.mean(np.stack(acts), axis=0)


def run_episode(env, init_state, instruction, url, model_name, max_steps, ensemble_n, video_path=None):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    ens = (VLA0Ensemble if model_name == "vla0" else SmolEnsemble)(ensemble_n)
    frames, lat, bad = [], [], 0
    for t in range(max_steps):
        out = query(url, obs, instruction)
        lat.append(out["latency_ms"]); bad += int(not out["well_formed"])
        act = ens.update(out["action"])
        act = np.array(act, dtype=np.float64)
        act[-1] = 1.0 if act[-1] > 0 else -1.0
        obs, reward, done, info = env.step(act.tolist())
        if video_path:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if done:
            break
    if video_path and frames:
        imageio.mimwrite(video_path, frames, fps=20)
    return bool(done), t + 1, float(np.mean(lat)), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8008")
    ap.add_argument("--label", required=True)
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--tasks", type=int, nargs="+", default=None, help="task ids per suite (default: all 10)")
    ap.add_argument("--episodes", type=int, default=2, help="init states per task")
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with urllib.request.urlopen(args.url + "/health", timeout=600) as r:
        health = json.loads(r.read())
    model_name = health["model"]
    print("policy:", health["engine"], model_name, health["quant"], flush=True)
    out_path = args.out or str(pathlib.Path(__file__).parent / "results" / "closed_loop" / f"{args.label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = dict(label=args.label, health=health, episodes=[], seed=args.seed, ensemble=args.ensemble)
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        results["health"] = health
    done_keys = {(e["suite"], e["task_id"], e["init_state"]) for e in results["episodes"]}
    t_start = time.time()
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        tids = args.tasks if args.tasks is not None else list(range(suite.n_tasks))
        for tid in tids:
            task = suite.get_task(tid)
            init_states = suite.get_task_init_states(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(args.seed)
            max_steps = args.max_steps or MAX_STEPS[suite_name]
            for ep in range(args.episodes):
                si = ep * (len(init_states) // args.episodes)
                if (suite_name, tid, si) in done_keys:
                    continue
                vp = None
                if args.video_dir:
                    os.makedirs(args.video_dir, exist_ok=True)
                    vp = os.path.join(args.video_dir, f"{args.label}__{suite_name}_t{tid:02d}_s{si:02d}.mp4")
                t0 = time.time()
                success, steps, lat, bad = run_episode(env, init_states[si], task.language, args.url, model_name, max_steps, args.ensemble, vp)
                results["episodes"].append(dict(suite=suite_name, task_id=tid, task=task.language, init_state=si, success=success,
                                                steps=steps, policy_latency_ms=lat, malformed=bad, wall_s=time.time() - t0))
                n = len(results["episodes"]); sr = np.mean([e["success"] for e in results["episodes"]])
                print(f"{suite_name:15s} t{tid:02d} s{si:02d} {'SUCCESS' if success else 'fail   '} steps={steps:3d} "
                      f"lat={lat:6.1f}ms  running SR={sr:.2f} ({n})  {time.time()-t_start:6.0f}s", flush=True)
                json.dump(results, open(out_path, "w"), indent=1)
            env.close()
    eps = results["episodes"]
    results["success_rate"] = float(np.mean([e["success"] for e in eps])) if eps else None
    results["per_suite"] = {s: float(np.mean([e["success"] for e in eps if e["suite"] == s])) for s in args.suites if any(e["suite"] == s for e in eps)}
    results["mean_policy_latency_ms"] = float(np.mean([e["policy_latency_ms"] for e in eps])) if eps else None
    json.dump(results, open(out_path, "w"), indent=1)
    print("success rate:", results["success_rate"], results["per_suite"], "->", out_path)


if __name__ == "__main__":
    main()
