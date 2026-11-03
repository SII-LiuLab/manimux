# Flexiv robot assembly

`FlexivRobot` uses the shared `RobotBase`, `EdgeRuntime`, session service,
recording and RoboGUI. There is no Flexiv-specific runtime. Independent arm
controllers own their RDK sessions; optional OmniGrippers own distinct USB2CAN
ports. Cross-device commands are not atomic.

## Models and adapters

Each `FlexivArm` owns only its seven-joint flange model. `RobotModel` composes
the separate tool's geometry and visual mapping into the complete group model
used by adapters and RoboGUI. Display placement and camera slots belong in the
RoboGUI preset; no Flexiv-specific dashboard or rendering adapter is needed.

Select `configs/embodiment/robot/flexiv_omnigripper_dual.yaml` for seven arm
radians and one normalized gripper coordinate per group (0 closed, 1 open).
The `flexiv_omnigripper` RoboGUI preset matches this layout. It shows arm geometry, aperture data and the locally installed Marvin
straight-gripper meshes with articulated fingers. The provisional gripper mounts
are configured per component; their calibration remains pending.

The RoboGUI display mounts are rotated 180 degrees about the world Y axis
through their shared midpoint. Left A is at `[0.45, 0.71, 1.72]` and right B at
`[-0.45, 0.71, 1.72]` m; the rotation reverses both horizontal and vertical
orientation without changing A/B identity. The stand and table remain fixed.
These are provisional display placements, independent of controller frames.

The joint and pose examples in `configs/experiments/example-flexiv/flexiv_omnigripper_*.yaml` use
existing `JointAdapter` and `PoseAdapter`, with the same robot model for decoding,
FK/IK and rendering. The pose template currently describes the flange reference;
it must not be represented as a calibrated grasp TCP. `scripts/validation/flexiv_offline.py`
checks both action contracts through the loader and wire codec without hardware.

`UmiDpTianjiAdapter` is not a Flexiv adapter: its fallback geometry, warmup joints,
differential IK and model-specific history contract cannot be copied by changing a
robot name. UMI deployment needs matching observations, checkpoint semantics and
Flexiv solver validation. No UMI checkpoint compatibility is claimed here.

## Recovery contract

Recovery is offered only with `execute: true`. Both controllers must be in
Auto (Remote) mode in Flexiv Elements. Ordinary Auto (`IN_AUTO_MODE`) and
Manual (`IN_MANUAL_MODE`) are rejected before hand-guiding enable, with the
robot serial and required operator action in the error. Enable timeouts report
the observed operational status. Station-specific Home targets belong
in the selected assembly's `home.joints_deg` (one seven-angle list per group).
There are no default hardware Home targets in the portable assembly. Configure:

```yaml
hardware:
  hand_guiding: true
  drag_stiffness_scale: 0.01
  hand_guiding_motion:
    control_hz: 100.0
    damping_ratio: 0.7
    max_velocity_deg_s: 6.0
    max_acceleration_deg_s2: 18.0
  home_motion:
    control_hz: 50.0
    peak_velocity_deg_s: 6.0
    tolerance_deg: 0.5
    settle_timeout_s: 5.0
```

- Clear error resets controller faults using temporary sessions, without enabling,
  homing, mode changes or gripper operations. Failure from either arm is reported.
- Home validates all arm targets against device limits before recovery enables any
  arm. It follows a cosine profile capped by the configured arm velocity, checks
  measured arrival, and leaves owned arm motion stopped in IDLE. RDK also limits
  velocity/acceleration internally. Once arm motion is stopped, Home sends opening
  1.0 to controlled grippers and waits for each measured opening to reach 0.98.
  Gripper arrival uses the same `settle_timeout_s` as arm settling; a timeout or
  motor fault is reported as Home failure. Connected Home respects
  `end_effector_control`; explicit idle Return Home always opens attached grippers,
  then restores the previous policy control setting and closes the assembly.
  The current low-stiffness gripper settings reached only about 0.966 in the
  isolated motion check, so full-open Home completion remains unverified on hardware.
  Cancellation,
  timeouts and faults stop and clean up owned components. Subsequent execution
  prepares NRT mode again. Fault clearing is explicit, not an automatic Home step.
- Drag selects A, B or AB and uses NRT_JOINT_IMPEDANCE with 1% nominal
  stiffness and damping ratio 0.7. Python refreshes the reference with the latest
  measured joint position at up to 100 Hz. There is no fixed entry-posture target.
  The RDK internally replans each target with the configured guiding velocity and
  acceleration limits; these limits are separate from policy and Home limits.
  Low limits can cause reference lag when the user moves faster than the internal
  trajectory generator. Higher limits may reduce that lag, but do not remove
  impedance stiffness, damping or establish physical acceptance.
- Release retains zero stiffness and monitoring only after its initial measured
  target. Release 5s starts its duration after confirmed entry. Guiding never
  homes, changes the active tool or rewrites payload settings. Verify the station's
  tool1 load in Elements. This Python NRT path needs no native floating extension.
- Guiding opens selected arm sessions only. All selected arms pass preflight
  before mode entry. Faults, stale feedback and unexpected modes abort guiding. Joint limits during
  manual Drag/Release are enforced by the Flexiv controller; ManiMux does not
  apply its additional policy target margin to hand guiding. Commanded trajectories
  retain their target-limit checks. Cancellation/lease loss closes owned sessions
  after blocking RDK Stop to IDLE. This is not servo-off or forced brake release.
  Grouped-RDK hand guiding is unsupported; the flat `flexivpy` backend is required.
- Native RDK calls can block; cancellation is observed between calls. A failed
  Stop retains ownership for retry. No bounded emergency-stop latency is promised.

The shared recovery service reports task state/errors; it does not publish a
continuous robot joint stream during idle recovery. Runtime measured-state display
is available during a prepared rollout. Do not confuse a demo or last displayed
pose with current feedback during recovery.

## Launch and verification

```bash
python -m manimux.robogui.dashboard --robot flexiv_omnigripper --demo
python scripts/validation/flexiv_offline.py --config manimux/configs/experiments/example-flexiv/flexiv_omnigripper_joint.yaml
python scripts/validation/flexiv_offline.py --config manimux/configs/experiments/example-flexiv/flexiv_omnigripper_pose.yaml
```

For live read-only feedback, omit `--demo` and run `manimux serve` with the
read-only example and the private station bindings. Prepare/Start connects real
hardware but does not issue targets. Its synthetic predictions are not measured
motion; never enable execution with the fake policy.

For recovery, use an executing real-policy configuration with the station's
assembly and bindings. The shared serving session exposes idle recovery before
Prepare, so Home/Drag do not need a model server. Prepare/Start require the actual
compatible policy service. Do not run another controller against the same arms.

Fake-SDK tests cover selection, capability gates, Home target preflight, trajectory
arrival/cancellation, fault reset, partial entry failure and retryable cleanup.
Offline joint/pose decoding produces eight coordinates per arm. Real Home/Drag,
combined motion and commanded OmniGripper operation remain unvalidated. Gripper mount/TCP calibration, controller load verification, sensors
and matching trained checkpoints remain installation requirements.

Station tuning (2026-10-28): the private assembly trials 30 deg/s and 90 deg/s²
for Drag reference following, compared with the portable defaults 6 and 18.
Stiffness, damping and feedback-following behavior are unchanged in this trial.
The official native torque floating implementation was withdrawn from the selected
path before physical acceptance. Real dragging feel and payload compensation
remain unverified. Fake SDK tests cover measured target updates, separate rate
limits, cancellation, faults and cleanup; they do not prove physical success.
