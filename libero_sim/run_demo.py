"""Watch pi0.5 control a simulated Franka arm in LIBERO (MuJoCo via robosuite).

A lighter-weight sibling of openpi/examples/libero/main.py. It runs a few chosen
tasks instead of the full 500-episode benchmark, lets you override the language
instruction, and writes videos showing both cameras with the instruction and
the outcome burned in.

The model does not run here. This process owns the simulator and sends each
observation (two 224x224 images, the 8-D proprio state and a prompt) to
./serve.sh over a websocket. It gets back a 10-step action chunk, of which
it executes the first `replan_steps` before asking again.

Runs under the Python 3.8 LIBERO venv; use ./run_demo.sh rather than calling
this directly.
"""

import collections
import dataclasses
import json
import logging
import pathlib
import time
from typing import List, Optional

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from robosuite.utils.transform_utils import quat2axisangle
import tyro

DUMMY_ACTION = [0.0] * 6 + [-1.0]  # hold still, gripper open
ENV_RESOLUTION = 256  # resolution the LIBERO training data was rendered at
MAX_STEPS = {  # longest training demo per suite, plus a margin
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    task_suite: str = "libero_goal"
    """libero_spatial | libero_object | libero_goal | libero_10 | libero_90"""
    task_ids: Optional[List[int]] = None
    """Task indices within the suite (default: the first three). --list-tasks prints them."""
    episodes: int = 1
    """Episodes per task, each starting from a different benchmark initial state."""
    prompt: Optional[str] = None
    """Override the task's instruction, e.g. to see how the policy reacts to a different command.
    LIBERO can only check the scene's original goal, so these runs are not scored."""
    list_tasks: bool = False
    show: bool = False
    """Open a live window while the arm runs (needs a display)."""
    replan_steps: int = 5
    num_steps_wait: int = 10
    """Idle steps at the start while dropped objects settle."""
    host: str = "localhost"
    port: int = 8000
    out_dir: str = "videos"
    seed: int = 7


def main(args: Args) -> None:
    suite = benchmark.get_benchmark_dict()[args.task_suite]()

    if args.list_tasks:
        for i in range(suite.n_tasks):
            print("%3d  %s" % (i, suite.get_task(i).language))
        return

    task_ids = args.task_ids if args.task_ids is not None else list(range(min(3, suite.n_tasks)))
    out_dir = pathlib.Path(args.out_dir) / args.task_suite
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to policy server: %s", client.get_server_metadata())

    results = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION, camera_widths=ENV_RESOLUTION)
        env.seed(args.seed)  # affects object placement even with a fixed initial state

        prompt = args.prompt or task.language
        for ep in range(args.episodes):
            logging.info("Task %d, episode %d: %r", task_id, ep, prompt)
            result, frames = run_episode(env, init_states[ep], prompt, client, args)
            result.update(task_id=task_id, episode=ep, task=task.language)
            results.append(result)

            tag = "unscored" if args.prompt else ("success" if result["success"] else "failure")
            path = out_dir / ("task%02d_ep%d_%s.mp4" % (task_id, ep, tag))
            imageio.mimwrite(path, frames, fps=20)
            logging.info("  -> %s in %d steps (%.1f ms/inference). Video: %s",
                         tag, result["steps"], result["ms_per_inference"], path)
        env.close()

    if args.prompt:
        print("\nCustom prompt: episodes are not scored, watch the videos in %s" % out_dir)
    else:
        successes = sum(r["success"] for r in results)
        print("\n%d/%d episodes succeeded" % (successes, len(results)))
        for r in results:
            print("  [%s] task %d ep %d: %s" % ("OK " if r["success"] else "FAIL", r["task_id"], r["episode"], r["prompt"]))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    if args.show:
        cv2.destroyAllWindows()


def run_episode(env, init_state, prompt, client, args):
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    frames = []
    inference_ms = []
    success = False
    max_steps = MAX_STEPS[args.task_suite]

    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    for t in range(max_steps):
        # LIBERO renders upside down relative to the training data; rotate 180 degrees.
        agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        if not plan:
            element = {
                "observation/image": _model_image(agent),
                "observation/wrist_image": _model_image(wrist),
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
                "prompt": prompt,
            }
            start = time.perf_counter()
            chunk = client.infer(element)["actions"]
            inference_ms.append((time.perf_counter() - start) * 1000)
            plan.extend(chunk[: args.replan_steps])

        action = plan.popleft()
        frame = _video_frame(agent, wrist, prompt, t, action)
        frames.append(frame)
        if args.show:
            cv2.imshow("pi0.5 in LIBERO", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            if not args.prompt:
                break

    if not args.prompt:
        frames.extend([_banner(frames[-1], success)] * 20)  # hold the last frame for 1 s
    return {
        "prompt": prompt,
        "success": success,
        "steps": t + 1,
        "inferences": len(inference_ms),
        "ms_per_inference": float(np.mean(inference_ms)),
    }, frames


def _model_image(img):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))


def _video_frame(agent, wrist, prompt, t, action):
    """Both cameras side by side at 2x, with the instruction and current action."""
    frame = np.concatenate([agent, wrist], axis=1)
    frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    frame = np.concatenate([frame, np.zeros((64, frame.shape[1], 3), np.uint8)], axis=0)
    h = frame.shape[0]
    cv2.putText(frame, prompt, (8, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    gripper = "close" if action[-1] > 0 else "open"
    info = "t=%3d  dpos=[%+.2f %+.2f %+.2f]  gripper=%s" % (t, action[0], action[1], action[2], gripper)
    cv2.putText(frame, info, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return frame


def _banner(frame, success):
    frame = frame.copy()
    text, color = ("SUCCESS", (60, 220, 60)) if success else ("FAILURE", (230, 60, 60))
    cv2.putText(frame, text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(tyro.cli(Args))
