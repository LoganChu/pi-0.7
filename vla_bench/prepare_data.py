"""Render the fixed observation set every engine is benchmarked on.

Runs under the LIBERO Python 3.8 venv (see ./prepare_data.sh). For each of the
40 LIBERO tasks (4 suites x 10) and a few benchmark initial states, it resets
the MuJoCo scene, lets objects settle for 10 no-op steps (exactly what VLA-0's
and openpi's eval loops do before the first policy query) and saves both
cameras, un-flipped, at the native 256x256 training resolution.

Outputs data/obs/<uid>_{agentview,wrist}.png and data/manifest.json.
"""
import argparse
import json
import os
import pathlib
import time

import math

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLE_STEPS = 10
RES = 256


def quat2axisangle(quat):
    """robosuite's quat2axisangle, as used by openpi/VLA-0 to build the 8-D LIBERO state."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def libero_state(obs):
    """eef pos (3) + eef axis-angle (3) + gripper qpos (2): the `state` field of the LeRobot LIBERO dataset."""
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)


def unflip(img):
    # LIBERO renders upside down; every LIBERO policy flips both axes back.
    return np.ascontiguousarray(img[::-1, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "data"))
    ap.add_argument("--init-states", type=int, nargs="+", default=[0, 17, 34])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "obs").mkdir(parents=True, exist_ok=True)
    observations = []
    t0 = time.time()
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
            env.seed(args.seed)
            init_states = suite.get_task_init_states(task_id)
            for si in args.init_states:
                env.reset()
                obs = env.set_init_state(init_states[si])
                for _ in range(SETTLE_STEPS):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                uid = "%s_t%02d_s%02d" % (suite_name, task_id, si)
                a_path, w_path = "obs/%s_agentview.png" % uid, "obs/%s_wrist.png" % uid
                imageio.imwrite(str(out / a_path), unflip(obs["agentview_image"]))
                imageio.imwrite(str(out / w_path), unflip(obs["robot0_eye_in_hand_image"]))
                observations.append(dict(
                    uid=uid, suite=suite_name, task_id=task_id, task_name=task.name,
                    instruction=task.language, init_state=si, step=SETTLE_STEPS,
                    state=[float(x) for x in libero_state(obs)],
                    agentview_path=a_path, wrist_path=w_path,
                ))
            env.close()
            print("%-16s task %2d  %-60s  %5.1fs" % (suite_name, task_id, task.language[:60], time.time() - t0), flush=True)

    manifest = dict(
        resolution=RES, settle_steps=SETTLE_STEPS, seed=args.seed,
        init_states=args.init_states, core_init_state=args.init_states[0],
        n=len(observations), observations=observations,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", len(observations), "observations to", out)


if __name__ == "__main__":
    main()
