# Guideline

[Project overview](../README.md) · [中文首页](../README.zh-CN.md) · [Documentation](README.md)

Run commands from the repository root. For your own installation of a supported robot,
start with [local station setup](../manimux/configs/local/README.md): fill one private
`manimux/configs/local/station.yaml` with actual CAN/IP/USB and service bindings.
The runtime, camera and Pi05 commands below read this file automatically. Use the same
`--local <path>` on each process to select another station.

## Hardware-free start

The project supports Python 3.11 and 3.12. Start an independent Viewer demo without a
checkpoint or hardware connection:

```bash
uv sync --dev
uv run manimux-viewer --robot yam --demo --port 8086
```

Open `http://127.0.0.1:8086`. The demo uses Viewer data without creating a robot connection.
Production robot entry points no longer include simulated robot drivers; test doubles
remain under `tests/`.

## Pi05 30k on YAM

This example uses the YAM hardware environment and the OpenPI model environment, with the
**pure-joint step-30000** checkpoint for `Put bottles into the bin.`. Prepare dependencies
and weights using the [Pi05 runbook](pi05-yam-runbook.md). Other models have their own
[deployment runbooks](README.md#policies-and-deployment).

Complete the [station guide](../manimux/configs/local/README.md), including
`paths.checkpoints`, and inspect the resolved configuration first.
The checkpoint's model identity and normalization must match this experiment.
Reuse matching camera/Viewer services when appropriate; collection and inference must
not control the same robot concurrently.

Run these in four separate terminals:

```bash
# Terminal 1: cameras
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml

# Terminal 2: Viewer (its network options remain independent)
envs/yam/.venv/bin/python -m manimux.viewer.dashboard \
  --robot yam --host 127.0.0.1 --port 8086

# Terminal 3: pure-joint 30k model server
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml

# Terminal 4: matching RTC runtime
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml
```

Open `http://127.0.0.1:8086`, then use **Prepare → Start rollout → Finish & Home**.
This experiment enables execution and moves to its configured start pose during Prepare.
The Viewer follows `policy.adapter.camera_map` reported by the runtime and labels model
inputs with their camera sources. Before receiving that mapping, it shows its labeled
default previews. For an explicit manual preview, use
`--config manimux/configs/viewer/yam-top.yaml`.
Normal rollouts have no scoring step. Experiment rollouts offer `Save evaluation` or
`Skip evaluation` before the next rollout. Skipping writes no human label.
See the [Viewer tutorial](viewer-tutorial.html) for controls.

The current recipe uses **`robot.control_hz: 30.0`**. Its model horizon is 50, action-point
spacing is `1/30 s`, and RTC `chunk_policy_steps` is 12. Twelve is the execution threshold for
requesting another chunk, not a truncation of the model's entire output; the old chunk
continues while inference finishes. This recipe does not interpolate 30 Hz model points
into a 100 Hz command stream.

The **joint+EE 30k** variant has a separate experiment and checkpoint contract:
`manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_ee_step30000.yaml`.
Read its runbook and ensure its checkpoint/stat subpaths exist below your local checkpoint
root. Select the matching experiment for every process; changing only one process does not
produce a matched deployment.

`policy backend identity mismatch` indicates that the runtime's expected model identity
and the server response disagree. Check the task, checkpoint, training configuration and
normalization rather than removing `expected_backend`. Adding `--check` to the Pi05 model
command checks paths and its contract without starting the service or the robot; it still
requires the model dependencies used to resolve that contract.

## Configuration and outputs

- `manimux/configs/local/station.yaml`: private devices, service addresses and local paths.
- `manimux/configs/experiments/<task>/<model>/`: robot, adapter, observation, inference, execution and recording.
- `manimux/configs/policy/<model>/<embodiment>/<task>/`: model/checkpoint and normalization contract.
- `manimux/configs/embodiment/robot/yam_control.yaml`: shared layout and motion limits for
  the paired Pi05 30k RTC recipes.

The control profile does not set model action spacing or force experiments
to use the same filtering or command frequency. Those choices remain explicit in each
experiment. Configuration changes do not update a running process; restart the affected
process for the next session. See the [configuration reference](../manimux/configs/README.md).

Inference writes session and rollout records under `run.output_dir`. Model targets, executor commands and achieved
feedback are different measurements; compare matching fields and timestamps.

Continue with [experiment design](experiment-design.md), [human feedback and recording](experiment-infra.md)
or [offline video evaluation](prm-as-a-judge.md).
