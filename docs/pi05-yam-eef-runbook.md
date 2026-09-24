# Pi05 joint+EEF checkpoint: YAM EEF execution

The paired experiment is
`manimux/configs/experiments/put_bottles/yam_pi05_manimux_eef_step30000.yaml`.
It uses the existing bottle-task joint+EEF step-30000 checkpoint. No retraining
or checkpoint modification is required.

## Action contract

The training targets have 26 dimensions: 14 joint/gripper values followed by
12 auxiliary EEF values. Each arm's six auxiliary values are translation and
rotation-vector deltas relative to the observation TCP frame. In EEF mode the
model adapter preserves all 26 values through the checkpoint's normalizer and
restores absolute targets using the observation's FK anchors:

- `p_target = p_observation + R_observation @ delta_position`
- `R_target = R_observation @ Rotation.from_rotvec(delta_rotation)`

Every row uses the same observation anchor; deltas are not integrated between
rows. Wire poses use XYZ+WXYZ. Grippers come from joint-output columns 6 and 13.
The model still receives joint state and the original three RGB cameras.

The runtime supplies the FK anchors, receives absolute EEF targets, and applies
YAM's existing Mink/quadprog IK with joint limits. Measured joints seed the first
solve; subsequent targets use the previous solution. The model predicts 50 rows;
only the first 20 rows enter IK and execution. Any failure rejects the chunk and
logs the arm, row, target, seed, gripper and solver failure reason. The grouped
IK interface does not expose rejected candidate joints or final residuals.
Successful plans retain the decoded EEF targets in adapter metadata.

This uses the existing ManiMux single-inflight scheduling recipe, action spacing
1/30 s and the existing YAM smooth executor. The EEF mode advertises only default
sampling; RTC, PAINT, AAC, AutoHorizon and DVAC are rejected. Existing joint recipes
continue to return 14 joint/gripper dimensions with their existing capabilities.

## Startup

From `/home/ubuntu/manimux`, use separate terminals. Finish the old rollout and
exit its runtime before starting a new robot owner. Matching camera and Viewer
services can be reused.

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/put_bottles/yam_pi05_manimux_eef_step30000.yaml
```

```bash
envs/yam/.venv/bin/python -m manimux.viewer.dashboard \
  --robot yam --host 127.0.0.1 --port 8086
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/put_bottles/yam_pi05_manimux_eef_step30000.yaml
```

```bash
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/put_bottles/yam_pi05_manimux_eef_step30000.yaml
```

The model endpoint is `ws://127.0.0.1:8530`. Viewer flow is Prepare, Start rollout,
Finish & Home. The experiment retains the existing initial/home pose behavior;
Prepare can move the arms. JAX compiles on the first request, so that first plan
may exceed the age budget and be rejected before subsequent warm requests.

## Verified scope

Seven focused tests passed, covering the training transform inverse, unchanged
joint outputs, supported sampling modes, the 20-row decode prefix and failure
diagnostics. A real-checkpoint GPU/WebSocket test using recorded observations
passed backend identity, reset, finite 50-row EEF outputs and all 24 IK solves
for an earlier 12-row prefix on both arms. The current 20-row prefix has offline
adapter coverage but has not repeated that real-checkpoint timing run. A warm
request measured 122 ms including transport/inference, then 89 ms of IK; the
cold request took 5.94 s to infer.
These are individual measurements, not latency distributions. Receipts are in
private `training/pi05-yam-eef-deploy/`.

No robot was moved during validation. The auxiliary EEF predictions have not
been evaluated for physical task success or for reachability from all start
poses. A passed recorded sample does not guarantee successful IK in a live run.

The runtime adapter uses grouped `RobotKinematics` for both injected and offline
construction paths. Regression checks cover both paths and replay the saved
real-model EEF targets through the same grouped FK/IK interface without hardware.
