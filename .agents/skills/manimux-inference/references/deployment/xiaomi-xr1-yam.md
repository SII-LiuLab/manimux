# Xiaomi XR-1 + YAM runbook

## Current local study

Use `manimux/configs/lists/xiaomi-xr1-bottles30k-ratio64-step-ik-all-tasks.yaml`
for five-task Serial/Async/RTC/PAINT comparisons. The RTC-only subset is
`manimux/configs/lists/xiaomi-xr1-bottles30k-rtc-beta10-step-ik-all-tasks.yaml`.
Both reference the same concrete variants. These installation-local studies are ignored by Git.
The former base-checkpoint and beta-4.25 launch entries were retired;
do not restore them as current experiment commands.

| Setting | Current study |
|---|---|
| Checkpoint | YAM put-bottles post-training, step 30000; not the mixed-task latest300ep checkpoint |
| Recipe | `manimux/configs/policy/xiaomi-xr1/yam/finetune-put-bottles-step30000.yaml` |
| Normalization | Checkpoint-paired `norm_stats.json` selected by that recipe |
| Native horizon / denoising | 30 actions / 5 denoising steps |
| Execution window | 18 source steps, explicitly selected for subsequent trials on 2026-10-20 |
| Action / robot rate | 30 Hz / 100 Hz |
| RTC | beta 10; initial delay estimated from warmup; 10-sample delay buffer |
| PAINT | 18 execution steps; initial delay 10; 10-sample delay buffer |
| Async | Progress-triggered, single in-flight request, zero blend |
| Serial | First step when ready, serial scheduling |
| IK | Step IK; 5 ms budget; no additional pose-speed limit |
| Rollout limit | 6000 control ticks, owned by each shared task YAML |
| Initial pose | Shared task default; moving cube uses raised_ready; Home remains selectable |
| Bottle evaluation | Four bottles; count/4; only 4/4 is full success; layouts four_01–four_10 |

Policy checkpoint/normalization, cameras, action contract and initial poses stay identical
across methods. Native model horizon and IK are model-specific; do not force a 50-step
horizon to make execution counts look equal. AutoHorizon is adaptive, not a fixed-window
method. BID and AutoHorizon are not in this step-IK study; do not silently change IK to expose them.

The bottle task is shared with the other comparison models. Its four-bottle rubric and
reference IDs are a new protocol, with separate variant identities and no imported
six-bottle scores. Historical recordings and reference images remain intact.

## Chunk IK comparison (2026-10-20)

For the operator-requested chunk IK trial, use
`manimux/configs/lists/xiaomi-xr1-bottles30k-ratio64-chunk-ik-all-tasks.yaml`.
Its 20 variants retain the step study's checkpoint, sampling, timing, scoring and
initial poses, with `execution.ik.mode: chunk`. Labels and recording variant IDs
explicitly say chunk IK; historical step IK records stay separate.
Chunk IK solves predicted trajectories during decoding instead of applying the
5 ms per-control-tick step IK budget. Decoding latency and convergence still need
warmup and operator validation. Config loading is not a physical trial.
The same model and camera services can be reused because their settings are unchanged.

Both IK lists and the RTC-only subset now use 18 source steps. The existing launch
filenames retain `ratio64` for command continuity, but the effective window is 18/30,
not the previous rounded 19/30. List IDs have a `steps18` suffix and recording
variants use `steps18` to keep subsequent trials separate from historical 19-step
results. Restart the runtime to load this change; the model's 30-action horizon
and five denoising steps are unchanged.

## Architecture and action contract

Model loading, preprocessing and sampling live in XPolicyLab's `Xiaomi_Robotics_1`
adapter using its vendored official Xiaomi source. ManiMux owns transport, action decoding,
FK/IK, scheduling, recording and RoboGUI; the model server does not control hardware.

The native output is a 30 × 60 sequence of observation-anchored end-effector deltas.
The YAM adapter reconstructs absolute poses and gripper openings against the request's
observation, clips normalized gripper openings to [0, 1], and step IK generates the
current joint command from measured joints. This is not native joint-position output.
RTC encodes the prior action condition into the same normalized model coordinates and
applies model-side guidance during denoising. An available sampler does not establish
real-robot task quality or acceptable end-to-end latency.

## Launch and validation

Use the same config list for model, camera and runtime; select methods/tasks in RoboGUI.
The model environment is `envs/xr1/.venv`, the camera/runtime environment is `envs/yam/.venv`.
The shared [user guide](../../../../../guideline.html) owns the launch workflow.
For an RTC-only run, use one terminal per command:

```bash
cd {repo}
envs/xr1/.venv/bin/python -m manimux.servers.policy \
  --config-list manimux/configs/lists/xiaomi-xr1-bottles30k-rtc-beta10-step-ik-all-tasks.yaml
```

```bash
cd {repo}
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --config-list manimux/configs/lists/xiaomi-xr1-bottles30k-rtc-beta10-step-ik-all-tasks.yaml
```

```bash
cd {repo}
.venv/bin/manimux-robogui --robot yam --host 0.0.0.0 --port 8086
```

```bash
cd {repo}
envs/yam/.venv/bin/python -m manimux run \
  --config-list manimux/configs/lists/xiaomi-xr1-bottles30k-rtc-beta10-step-ik-all-tasks.yaml
```

Only the operator performs Prepare, Warmup, Start and Finish. Do not restart an active
rollout to apply configuration changes. Restart the idle runtime to reload its list.
Before giving commands, run `manimux inspect --config-list ...` and each service's
`--check`; these checks do not open devices or establish hardware readiness.

Records resolve under `data/experiments/<list-id>/<task>/<variant>/`.
The 2026-10-19 configuration cleanup preserves all previous recordings. Its source
backups and loader audit are in the local `data/preparation/put-bottles-four-v1/` directory.
