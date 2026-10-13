# Pi05 + YAM runbook

> 2026-10-09: The old screwdriver, red-object, pass-ball and offline experiment entries below were retired. These commands are historical; select a current entry from the [experiment index](../../../../../manimux/configs/experiments/README.md). Diagnostic tools now require an explicit experiment argument.


This guide covers the Pi05 model, checkpoints, input/output contracts, and startup commands. Inference-method principles,
complete parameters, and validation steps belong in their respective method documents rather than being duplicated here.

## Bottle placement step-30000

Joint-only and joint+EE use different checkpoints. Even though both use port `8500`, their
server and infrastructure configurations cannot be mixed. Run the following commands from the repository root with YAM / OpenPI environments installed and
the configured checkpoints and cameras prepared. README commands do not automatically install weights or local device configurations.

### Joint-only + RTC

The complete four-terminal startup sequence is in the [user guide](../../../../../guideline.html#launch):
cameras → RoboGUI → `server/put-bottles/joint-step30000.yaml` model service
→ `infra/put-bottles/rtc-joint-step30000.yaml` runtime.

This pairing uses `pi05-yam-put-bottles-joint-step30000` weights, model horizon 50, trajectory interval
`1/30 s`, and RTC `chunk_policy_steps` 12. This does not truncate the model output to 12 steps:
after RTC requests the next inference, the old chunk keeps executing while waiting for the response.

### Switch to joint+EE

Stop the old model server and current runtime, then replace the configurations in both commands with:

- server: `manimux/configs/policy/pi05/yam/put-bottles/joint-ee-step30000.yaml`;
- runtime: `manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_ee_step30000.yaml`.

Both RTC configurations explicitly reference [`manimux/configs/embodiment/robot/yam_control.yaml`](../../../../../manimux/configs/embodiment/robot/yam_control.yaml),
sharing hardware parameters, action intervals, and motion limits with YAM data collection. Execution remains 100 Hz Smooth with 8 Hz filtering.
The shared configuration currently adds no arm velocity/acceleration limits; the gripper retains a `1.0 /s` closing-target limit and switches opening targets directly.
This does not remove hardware protections or guarantee that physical closure takes exactly one second.
See [shared control notes](../../../../../manimux/configs/README.md#arm-motion-limiting). Other historical configurations do not inherit these settings automatically.

For `policy backend identity mismatch`, first check that the model actually loaded on the port matches the configuration.
Do not remove `expected_backend` to bypass validation. To check local paths and contracts without starting a service,
add `--check` to the model-server command.

## Local red-ball task checkpoint

This training initialized from official `pi05_base` and fine-tuned on these 20 YAM episodes:

```text
/home/ubuntu/yam-abc-reproduce/data/episodes/pick_the_red_ball_up_and_place_it_into_the_box/
```

Inference uses the matching stats bundled with the step-1000 checkpoint, not Robocurve stats:

```text
checkpoints/finetuned/ziyang/pi05-yam-pick-red-ball-box-b384/1000/
  params/
  assets/yam_pick_red_ball_box_v1/norm_stats.json
```

- server: `manimux/configs/policy/pi05/yam/finetune-pick-red-ball-box-step1000.yaml`;
- ManiMux: `manimux/configs/experiments/pick_red_object/pi05/yam_pi05_manimux_step1000.yaml`;
- Input: three separate RGB views, 14D YAM state, and red-ball task text.
- Output: `50 x 14` absolute joint positions.
- Timing: trajectory points at 30 Hz, low-level commands at 100 Hz.
- Initial pose: first frame of episode `20260812_193716_0751770e`, matching both arms' joint and gripper values exactly.
- Evidence: offline GPU inference with the real checkpoint, XPolicy WebSocket, three cameras, ManiMux, and dual-arm
  YAM hardware integration all passed. The model visibly reproduces demonstrations, but the 20 practice episodes have limited quality and cannot reliably
  complete the task. Record this as successful deployment with limited policy quality, not as a task success rate.

Check paths and contracts only:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  manimux/servers/pi05.py --check \
  --config manimux/configs/policy/pi05/yam/finetune-pick-red-ball-box-step1000.yaml
```

Offline GPU forward inference uses the 14D state from that episode's first frame, without cameras, CAN, or a robot:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/validation/pi05_yam_offline_infer.py \
  --config manimux/configs/policy/pi05/yam/finetune-pick-red-ball-box-step1000.yaml \
  --infra-config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_manimux_step1000.yaml
```

Model service:

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  manimux/servers/pi05.py \
  --config manimux/configs/policy/pi05/yam/finetune-pick-red-ball-box-step1000.yaml
```

After camera, CAN, and preflight checks, the operator runs ManiMux on hardware:

```bash
envs/yam/.venv/bin/manimux run \
  --config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_manimux_step1000.yaml
```

## Screwdriver step-15000: seven algorithms, one executor

These configurations share the step-15000 checkpoint, checkpoint-matched norm stats, task text,
three-camera mapping, initial pose, and Recorder. Native output is always `50 × 14` absolute joint actions,
with trajectory interval `1/30 s`. Algorithms alter sampling, chunk selection, and scheduling, not the low-level executor.

All configurations live under `manimux/configs/experiments/assemble_screwdriver/pi05/`, named
`yam_pi05_<algorithm>_step15000.yaml` (algorithm names use underscores):

| Algorithm / filename prefix | Runtime | Algorithm configuration |
|---|---|---|
| `manimux` | `manimux` | Single-inflight, refill at 0.4 s remaining, blend four steps |
| `rtc` | `rtc` | Minimum execution 20 steps, initial delay four steps, beta 9.1, blend four steps |
| `act-temporal-ensemble` | `act_temporal_ensemble` | Coefficient 0.01, request every 20 policy steps (approximately 0.667 s) |
| `aac` | `aac` | 20 candidates, motion threshold 0.2, backward beta 0.99 |
| `paint` | `paint` | Execution steps 12, initial delay 10 steps, delay history window 10 |
| `autohorizon` | `autohorizon` | Integrated JAX selector; the model returns the execution length |

Shared low-level settings: `executor.type: smooth`, 100 Hz control, 8 Hz cutoff, joint velocity limit
`0.25 rad/s`, acceleration limit `0.5 rad/s²`, absolute position limit `3.14 rad`; both grippers use
continuous `0–1`, velocity limit `1.0 /s`, and acceleration limit `12.0 /s²`.
These match the original screwdriver Async / RTC settings. ACT, AAC, PAINT, and AutoHorizon retain
`blend_policy_steps: 0` per their contracts; blending belongs to Timeline stitching, not SmoothExecutor.

AAC retains existing `yam_60ep_ee_increment.json` EE increment statistics **for candidate scoring**.
These were not re-estimated specifically for the screwdriver task and do not replace checkpoint action normalization.
Each method retains its existing cold-start timeout, with outputs written separately under
`data/experiments/pi05-assemble-screwdriver-step15000/<algorithm-prefix>/`.

ACT, AAC, PAINT, and AutoHorizon screwdriver configurations were added on 2026-09-08.
This reuses existing algorithm implementations and passes configuration/unit regression checks; it does not mean these four new combinations have completed GPU
or hardware tests with this checkpoint. See method documents below for existing measured evidence. Historical red-ball and Robocurve configurations remain unchanged.

First start the shared Pi05 policy server from `/home/ubuntu/manimux`:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  manimux/servers/pi05.py \
  --config manimux/configs/policy/pi05/yam/finetune-assemble-screwdriver-step15000.yaml
```

The model service need not change with the algorithm. Select a runtime in another terminal, for example PAINT:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/assemble_screwdriver/pi05/yam_pi05_paint_step15000.yaml
```

Replace the filename prefix to select another algorithm from the table, for example RTC:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/assemble_screwdriver/pi05/yam_pi05_rtc_step15000.yaml
```

Run one of the six runtimes at a time, all sharing the policy server above.

Training-free Pi05 inference methods are maintained in their own method documents:

- ACT temporal ensemble: [`act-temporal-ensemble.md`](../inference.md);
- AAC: [`reproductions/aac-pi05.md`](../reproductions/aac-pi05.md);
- PAINT: [`reproductions/paint-pi05.md`](../reproductions/paint-pi05.md);
- AutoHorizon: [`reproductions/autohorizon-pi05.md`](../reproductions/autohorizon-pi05.md).

The following sections record separate experiments with the earlier Robocurve 16-step checkpoint; do not mix them with local 50-step configurations.

## Robocurve model

YAM fine-tuned weights are located at:

```text
checkpoints/finetuned/robocurve/pi05-yam-molmoact2/  # robocurve/pi05-yam-molmoact2
  params/
  assets/yam-bimanual-merged/norm_stats.json
```

- Input: three separate RGB views (base, left wrist, right wrist), not a stitched image.
- State: 14D, left `6+1` followed by right `6+1`.
- Output: `16 x 14` absolute joint positions, 30 Hz.
- Flow sampling: OpenPI default 10 steps.
- Stats: checkpoint-bundled quantile norm stats.
- Release source: `robocurve/pi05-yam-molmoact2`.
- server: `manimux/configs/policy/pi05/yam/finetune.yaml`;
- ManiMux 30 Hz default configuration: `manimux/configs/experiments/pick_red_object/pi05/yam_pi05_manimux.yaml`.
- Pi-guided RTC comparison: `manimux/configs/experiments/pick_red_object/pi05/yam_pi05_rtc.yaml`.
- 50 ms stretched-timing comparison: `manimux/configs/experiments/pick_red_object/pi05/yam_pi05_stretched_50ms.yaml`.

## Recorded validation status

One actual-checkpoint GPU forward pass without hardware completed, producing finite `16 x 14` absolute
joint actions. A real YAM rollout on 2026-08-20 showed clear movement toward the pick task, but the old configuration's
`0.20 rad/s`, `0.40 rad/s²` low-speed envelope caused slow tracking and jitter,
so this trial cannot establish a success rate.

The episode is located at:

```text
data/archive/pre-campaign-20260824/root-runs/run-20260820T061701Z-8cfd2e00/episode-6bcee9a699eb/
```

The recording contains 143 inference submissions and 141 accepted plans; typical stale prefixes were only 2–3 steps, showing the 16-step
horizon was not exhausted by inference latency. The documented ManiMux infrastructure uses shared `smooth` executor
parameters: 8 Hz cutoff, `0.25 rad/s`, `0.5 rad/s²`. Pi05 retains only the model contract's 30 Hz
and 16-step horizon. These executor values are tracking
limits, not thresholds judging model-action quality.

## 1. Offline checks

Check paths and contracts without loading the model:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  manimux/servers/pi05.py --check \
  --config manimux/configs/policy/pi05/yam/finetune.yaml
```

Run GPU forward inference on synthetic three-camera observations without starting a service or robot:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/validation/pi05_yam_offline_infer.py \
  --config manimux/configs/policy/pi05/yam/finetune.yaml
```

## 2. Model service

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  manimux/servers/pi05.py \
  --config manimux/configs/policy/pi05/yam/finetune.yaml
```

A foreground terminal without continuous logs is normal. Check readiness:

```bash
ss -ltnp | rg ':8500'
nvidia-smi
```

`127.0.0.1:8500` must be in `LISTEN` state.

<a id="3-robogui"></a>

## 3. Cameras and RoboGUI

Camera service:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml
```

Do not duplicate an existing `5555` service. RoboGUI is optional:

```bash
envs/yam/.venv/bin/manimux-robogui --robot yam --host 0.0.0.0 --port 8086
```

## 4. Preflight

Read the three real cameras and current joints and run inference, while explicitly disabling start-position movement and Home on exit, and sending no
model actions:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/pi05_base_yam_preflight.py \
  --config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_manimux.yaml
```

Save the printed measured state, first action, shape, gripper range, and steady-state latency. The script
reports contracts only, without a subjective `safe_for_live` conclusion.

## 5. Hardware runtime

Clear the workspace, keep the emergency stop ready, and confirm both CAN interfaces report `ERROR-ACTIVE`:

```bash
for c in can_left can_right; do
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

The default ManiMux experiment retains the Pi05-YAM checkpoint's 30 Hz timing:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_manimux.yaml
```

This ManiMux baseline executes 16 absolute joint points over approximately 0.50 seconds. The robot control loop also runs at 30 Hz to preserve
the checkpoint's learned timing semantics; asynchronous scheduling, Timeline, SmoothExecutor, and recording remain
ManiMux responsibilities.

The 50 ms stretched comparison executes the same points over approximately 0.75 seconds:

```bash
envs/yam/.venv/bin/manimux run --config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_stretched_50ms.yaml
```

These are separate comparison experiments; do not run them concurrently. In the actual 50 ms trial on 2026-08-20, model requests and
plan acceptance were normal, but replanning was visibly slower and the operator observed a forward twitch followed by a return. Thus 50 ms
is retained only as a failed comparison, not the default hardware configuration.

After connection, the robot takes 3.5 seconds to reach the YAML start pose; this is not a model action. The documented executor is much faster than the initial hardware rollout,
so keep the first trial with new parameters short and the emergency stop ready.

## Chunk-boundary diagnostics

The ordinary runtime writes a `plan_boundary` event to the episode's
`events.jsonl` for each accepted plan. It records the first raw model point, first ManiMux-stitched point, previous command, and measured
joints. This adds diagnostics without altering actions, speeds, or stitching. Even after `Ctrl-C`, records remain in the
`.partial` episode.

After short rollouts of two policies to compare, analysis automatically selects the latest valid episodes for different policies.
Use `--episode` to select recordings explicitly, or inspect them offline:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/analyze_chunk_boundaries.py
```

## Pi05 YAM finetune + RTC

The step-1000 red-ball-in-box experiment uses the same fine-tuning configuration; no second weight instance is required:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  manimux/servers/pi05.py \
  --config manimux/configs/policy/pi05/yam/finetune-pick-red-ball-box-step1000.yaml
```

RTC uses a separate infrastructure configuration. For repeated evaluation, start one persistent session service:

```bash
envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_rtc_step1000.yaml
```

It shares the step-1000 Default configuration's `50 x 14` checkpoint contract, 100 Hz robot loop,
30 Hz policy points, `0.25 rad/s`, `0.50 rad/s²`, and three-second start/Home. The RTC pilot uses
`min_execute_policy_steps: 20`, `initial_delay_policy_steps: 4`, and `beta: 9.1`; runtime updates the delay forecast
from actual round-trip measurements. Only RTC scheduling and Pi-guided denoising conditions change.

Each run creates:

```text
data/experiments/pi05-red-ball-box-step1000/rtc/session-*/rollout-*/
```

After an episode is finalized, RoboGUI enables `Task result`, failure tags, and notes,
saved to `rollout-*/evaluation/human-label.json`. Task results are separate from `success` in `result.json`, which indicates
normal runtime finalization.

RoboGUI selects modes via Prepare normal / Prepare experiment. Normal mode omits scoring; after an experiment,
save a manual evaluation or click `Skip evaluation` to continue. Skipping creates no scoring file.
For formal experiments, select the Task and position `01`–`10` in the Top reference layout, plus `Experiment repeat=1/2/3`.
Prepare fixes the position, repeat, and reference-image path/hash. The displayed task command is actually sent to Pi05, not merely shown as text.

`serve` neither loads the model nor starts cameras or replaces RoboGUI. Start the camera server, Pi05 model
server, and `manimux-robogui` separately, then start `serve` once. After RoboGUI reports service ready:

1. Confirm the task; for formal experiments also choose the reference image and repeat.
2. Click `Prepare normal rollout` or `Prepare experiment rollout`; ManiMux creates a fresh episode, connects the robot, and moves to the start pose.
3. Wait for `PAUSED`, check the physical robot, then click `Start rollout`.
4. Use `Pause / Hold` to hold position and `Resume rollout` to continue.
5. On completion or failure, click `Finish & Home` and wait for Recorder finalization and robot homing.
6. With experiment mode ON, fill in and save the manual evaluation; after the service returns to idle, click the next `Prepare new rollout`.

Every formal configuration asynchronously saves each camera's MP4 at 10 Hz without blocking the control loop. Afterward, check
`dropped_bundles` and `error` in `videos/index.json`. For the complete evidence layout, see
[experiment infrastructure](../../../manimux-experiments/references/records.md).

Each episode creates a new worker session and resets Timeline, RTC delay history, Executor, Recorder, and
RoboGUI trails, without carrying inference state from the previous rollout. Camera/model/robogui/service processes stay running.

For a single rollout or script compatibility, the original CLI entry point remains available:

```bash
envs/yam/.venv/bin/manimux run \
  --config manimux/configs/experiments/pick_red_object/pi05/yam_pi05_rtc_step1000.yaml
```

## Shutdown

For a single `run`, press `Ctrl-C` once in its foreground terminal and wait for Home and exit. For persistent `serve`, first finish the current
rollout in RoboGUI; after service idle, press `Ctrl-C` in the serve terminal. Only then stop RoboGUI, model, and cameras.
Do not use broad `pkill` commands.

## Pi05 base + RTC

Base zero-shot is a separate experiment:

```text
server: manimux/configs/policy/pi05/yam/base.yaml
infra:  manimux/configs/experiments/pick_red_object/pi05/yam_pi05_base_rtc.yaml
```

It uses 50-step Pi-guided RTC and does not represent the default YAM fine-tuned configuration. Do not overwrite this guide's
16-step checkpoint experiment with the base configuration. Base weights remain at `checkpoints/pretrained/pi05/pi05_base`; it only
reuses norm stats from the Robocurve YAM checkpoint, so the configuration prints
`checkpoint_source` and `norm_stats_source` separately.
