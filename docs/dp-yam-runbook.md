# Ordinary DP on YAM: bottle task, absolute EEF

The paired experiment is
`manimux/configs/experiments/put_bottles/dp/yam_dp_manimux_eef_step100000.yaml`.
It uses the ordinary XPolicyLab DP model, checkpoint EMA weights, 100 DDPM
sampling steps, three RGB cameras, three measured observations at 30 Hz,
and six action waypoints per request. The training horizon is eight: the model
selects rows 2 through 7 because the first three observation frames end at row 2.
Actions are absolute `grasp_site` poses in each arm's own base frame. Model
coordinates are XYZ metres and fixed-axis XYZ Euler radians; wire poses are
XYZ+WXYZ. Gripper is normalized, zero closed and one open.

## Installation and checkpoint

From the repository root:

```bash
bash XPolicyLab/policy/DP/install_inference.sh
```

Expected local checkpoint:
`checkpoints/finetuned/ziyang/dp-yam-eef-put-bottles-step100000/100000.ckpt`.
SHA-256: `579abd982af3289bd6835112bebe197a0e9c81f1403b6d9d8573326a48c30534`.
The server verifies this digest before loading. Normalization and architecture
come from the checkpoint. The model environment contains torch; the robot
runtime uses `envs/yam/.venv` and does not import model code.

## Start

Run from the repository root, one terminal per process. Reuse existing camera
and Viewer processes when their configuration matches.

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/put_bottles/dp/yam_dp_manimux_eef_step100000.yaml
```

```bash
envs/yam/.venv/bin/python -m manimux.viewer.dashboard \
  --robot yam --host 127.0.0.1 --port 8086 \
  --config manimux/configs/viewer/yam-dp-live.yaml
```

```bash
XPolicyLab/policy/DP/.venv/bin/python -m manimux.servers.dp \
  --experiment manimux/configs/experiments/put_bottles/dp/yam_dp_manimux_eef_step100000.yaml
```

```bash
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/put_bottles/dp/yam_dp_manimux_eef_step100000.yaml
```

The policy endpoint is `ws://127.0.0.1:8520`; a private station may override it
with a `policy_dp` service. The existing station supplies CAN and camera serial
bindings. Viewer flow is Prepare, Start rollout, Finish & Home. The experiment
enables real execution and retains the YAM start/home behavior of the existing
bottle experiment; Prepare may move the arms.

The history decorator waits for three distinct measured camera/state samples
approximately 33 ms apart. It never fills history with repeated polling frames.
The Viewer configuration displays current physical camera frames. DP's `_t0`,
`_t1`, and `_t2` model inputs are temporal aliases assembled separately by the
adapter and are not physical stream names published to the Viewer.
The experiment uses the standard ManiMux defaults: deadline scheduling, a
0.4-second refill threshold, 0.02-second commit lead and two blending steps.
Inference may overlap execution; the model uses ordinary DDPM sampling, without
RTC guidance. The six-step chunk is shorter than the observed inference plus IK
latency, so overlapping requests does not guarantee uninterrupted motion.
Unreachable IK targets reject the chunk. Recorded evidence goes under
`data/experiments/dp-put-bottles-eef-step100000/manimux-default`.

## Evidence boundary

On 2026-09-24, the downloaded checkpoint matched the remote SHA-256. Three
GPU forwards on recorded input passed (521 ms cold, 251/227 ms warm). A separate
WebSocket test using the actual runtime adapter passed backend identity, reset,
and finite action checks; all twelve IK targets succeeded, producing two 6x7
joint chunks. That request took 453 ms including transport/inference, followed
by 119 ms of IK. These are individual offline measurements, not a latency
distribution or a guarantee under live camera/control load. Test receipts are
in the private `training/dp-yam-eef-deploy/` directory.

Offline pose, history and transport tests do not establish physical bottle-pick
success. Hardware motion and task success require a separate real rollout.
