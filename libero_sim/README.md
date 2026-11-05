# libero_sim — π₀.5 controlling a simulated robot arm

The companion to `fastlab` and `memlab`. Those rebuild pieces of π₀.x to show
how they work; this directory runs the **real, open-weight π₀.5** from
Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi)
on a 7-DoF Franka Panda arm in
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), a MuJoCo
manipulation benchmark built on robosuite.

```
  run_demo.py  (Python 3.8, CPU + EGL)          serve.sh  (Python 3.11, JAX, GPU)
 ┌──────────────────────────────────┐          ┌──────────────────────────────────┐
 │ MuJoCo / robosuite / LIBERO      │  obs     │ π₀.5  (pi05_libero checkpoint)   │
 │  agent cam 224² + wrist cam 224² │ ───────► │  PaliGemma VLM + action expert   │
 │  eef pos, axis-angle, gripper    │  prompt  │  flow matching → 10×7 chunk      │
 │                                  │ ◄─────── │                                  │
 │ execute 5 actions, then replan   │ actions  │                                  │
 └──────────────────────────────────┘ websocket└──────────────────────────────────┘
```

There are two processes because LIBERO is pinned to Python 3.8 and old
robosuite, while openpi needs Python 3.11 and JAX. openpi's own example is
split the same way.

Each action is 7-D: an end-effector position delta, a rotation delta
(axis-angle), and a gripper command. The checkpoint is π₀.5 fine-tuned on all
four LIBERO suites. openpi reports about 97% average success.

## Setup (once)

Requirements: Linux, an NVIDIA GPU with at least 8 GB free for inference, `uv`,
and about 15 GB of disk for the checkpoint.

```bash
./setup.sh
```

This clones openpi at a tested commit into `openpi/` (gitignored), builds both
virtual envs, and writes a LIBERO config so its first import doesn't stop to
ask questions.

## Run

```bash
# terminal 1 — first run downloads the 11.6 GB checkpoint to ~/.cache/openpi
./serve.sh

# terminal 2
./run_demo.sh                                   # first 3 libero_goal tasks, 1 episode each
./run_demo.sh --list-tasks                      # list the tasks in a suite
./run_demo.sh --task-suite libero_10 --task-ids 0 4 --episodes 3
./run_demo.sh --task-ids 0 --prompt "open the top drawer of the cabinet"   # give it a different command
./run_demo.sh --show                            # also open a live window
```

Videos go to `videos/<suite>/taskNN_epN_{success,failure}.mp4`. Each frame
shows the agent-view camera on the left, the wrist camera on the right, and the
instruction and current action underneath. A run summary is written to
`summary.json`.

To try the autoregressive **π₀-FAST** model instead (it outputs FAST tokens,
the same tokenizer `fastlab` reimplements):

```bash
./serve.sh policy:checkpoint --policy.config pi0_fast_libero --policy.dir gs://openpi-assets/checkpoints/pi0_fast_libero
```

## Things to try

- **Instruction following.** `libero_goal` puts the same objects in the same
  scene, so only the language says what to do. Use `--prompt` to give one
  task's scene another task's instruction. For example,
  `--task-ids 0 --prompt "put the bowl on the plate"` works. These runs are
  unscored because LIBERO can only check the scene's original goal, so watch
  the video.
- **Long horizon.** `libero_10` tasks chain two sub-goals. This is where π₀.5's
  failures tend to show up.
- **Replanning rate.** `--replan-steps 1` runs closed-loop on every step.
  `--replan-steps 10` runs the whole chunk open-loop.

## Troubleshooting

- `EGL` errors: run with `MUJOCO_GL=glx ./run_demo.sh ...`. This needs a
  display.
- If the client can't connect, wait for `serve.sh` to log that it is serving on
  port 8000. Loading the model and the first JIT compile take about a minute.
