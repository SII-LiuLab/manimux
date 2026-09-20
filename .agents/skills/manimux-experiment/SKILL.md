---
name: manimux-experiment
description: Select paired ManiMux policy-server and runtime configs, explain service roles, provide full startup commands and operate requested robot experiments. Use for inference, collection startup and runtime-mode changes on an already specified station.
---

# ManiMux Experiment

Paths and commands are relative to the ManiMux root. Establish the requested body, task,
checkpoint, joint/EE action variant and runtime from the request and selected files.
If local hardware binding is missing, use `manimux-station-setup`; do not guess serials.

## Services

| Process | Responsibility | Host |
|---|---|---|
| Camera server | Opens configured cameras and provides frames; usual REP/PUB ports are 5555/5556 | Camera-connected machine |
| Policy server | Loads the checkpoint and predicts actions; no robot or camera ownership | Compatible local or remote model environment |
| `manimux serve` | Owns the robot driver, chunk scheduling and executor | Machine with CAN/controller access |
| Viewer | Displays observations/trajectories/chunks and sends experiment requests | Reachable by the runtime and browser |

Use configured endpoints, not the default port as proof of which model is running.
Camera history currently converts host wall-clock timestamps to monotonic time; a remote
camera server requires clock alignment, not just an edited endpoint.

## Choose a matching pair

Read the selected experiment, its `policy_server` recipe and its `control_profile`.
Match task, checkpoint, action representation, normalization and backend identity.
A mismatch usually means the wrong server/config pair; do not remove `expected_backend`.
Model horizon, action interval, execution frequency, `chunk_steps` and chunk truncation
are different settings. Read the chosen strategy before promising how many actions execute.
RTC/PAINT selection also requires the corresponding model sampler support.

The existing Pi05 pure-joint 30k example, assuming its local environments, devices and
checkpoint paths have been prepared, is four separate terminals:

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config manimux/configs/embodiment/sensor/cameras/yam.yaml
```
```bash
envs/yam/.venv/bin/python -m manimux.viewer.dashboard --robot yam --host 127.0.0.1 --port 8086
```
```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --config manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml
```
```bash
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/put_bottles/yam_pi05_rtc_joint_step30000.yaml
```

Read `docs/guideline.md` and the chosen model/body runbook for another recipe. These are
examples, not commands to run automatically. For Tianji UMI_DP, follow
`docs/umi_dp-tianji-runbook.md`: checked-in runtime templates are unbound until the
launcher binds a real checkpoint to a paired server/runtime configuration.

## Run only the requested workflow

- Reuse matching existing camera/Viewer services. If the user asks only for commands,
  return commands; do not launch them. If asked to run, do not start a competing robot
  owner or kill unrelated collection, model or training processes.
- Viewer flow is Prepare → Start rollout → Finish & Home. `manimux/session.py`
  waits for Prepare before creating a rollout; `manimux/runtime/edge.py` connects
  its driver during startup. Prepare can move YAM when the chosen config enables it.
  Inspect the selected runtime for paused inference behavior rather than assuming that
  inference and command execution start together.
- For collection, use the body's own entry point. YAM uses
  `python -m manimux.collection --config manimux/configs/collection/yam/station.yaml` from its
  hardware environment; `--mock` selects a hardware-free GUI. Starting teleop includes
  alignment and following, not merely opening a connection. See `docs/yam-collection.md`.
- Report which config/endpoint is running and the actual session/episode output directory.
  Distinguish configuration inspection, offline forward, server readiness and robot execution.
  A completed rollout is not itself evidence that the manipulation task succeeded.
