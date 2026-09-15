# UMI DP on Tianji

The model is implemented in `XPolicyLab/policy/UMI_DP`, with vendored model and
training source. ManiMux handles camera history, FK/IK and execution through the
existing `xpolicylab_ws` worker. The unified runtime loop is unchanged.

## Prepare and bind a checkpoint

Run from the ManiMux root. The model uses its own Python 3.11 venv; the hardware
runtime needs the normal Tianji dependencies plus ManiMux's `xpolicylab` extra.

```bash
bash XPolicyLab/policy/UMI_DP/install.sh envs/umi_dp/.venv
# Check the actual EMA/model, SHA, H, dt, first offset, and preprocessing:
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --checkpoint /path/to/trusted/pass_ball.ckpt --check
# Write a paired model-server and runtime config, bound to those artifacts:
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --checkpoint /path/to/trusted/pass_ball.ckpt \
  --bind-runtime-config data/experiments/pass-ball-bound.yaml
# Select RTC with --runtime-template configs/umi_dp/tianji/infra/pass_ball/rtc.yaml.
```

The binder updates H16/H64 and observation/action timing from the checkpoint,
checks the shared control profile, and refuses to overwrite outputs. The model
checks the paired `expected_artifacts`; ManiMux verifies the server's identity
and sampling capabilities before execution. The checked-in templates are
intentionally unbound and rejected by the actual Tianji adapter until bound.

`configs/robots/tianji/common.yaml` owns all device/command limits and has 30Hz
action points. A legacy 100ms-action checkpoint cannot bind to that profile.
To use one, explicitly create/select a matching control profile and runtime
template with its action interval, retaining reviewed hardware and motion limits;
the binder will never silently change those settings. Its first action offset
is still 1/source_fps (33.3ms at 30fps).

## Start the model service

```bash
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --config data/experiments/pass-ball-bound-server.yaml
```

Use the model README's `EVAL_ENV_TYPE=debug` commands to exercise real weights
without devices. The dedicated debug client checks plain and encoded RGB,
absolute EE action shapes, real RTC conditioning, batches and reset. It does not
provide evidence of robot motion or task success.

The hardware-side configuration uses the existing camera server's PUB endpoint
(default `tcp://127.0.0.1:5556`) and configured wrist names `left_wrist` and
`right_wrist`. Start camera/runtime services only as part of the intended hardware
session, using the existing Tianji camera config and the paired runtime config.
The common Tianji profile remains `execute: false`, `gripper_control: false` by
default. No devices are accessed by model validation or binding.

## Viewer: manual drag and direct homing

Use `manimux serve --config <runtime-config>` with the Tianji viewer. The
**Manual recovery** panel is always visible below the task heading, independent
of preparation, execution, interruption or evaluation:

- **Finish & Home** saves the current rollout and returns home directly.
- **Finish without homing** saves and releases the robot at its current pose,
  overriding `robot.options.home_on_close` for that finish request.
- During an active rollout, select **Drag arms: A / B / AB** and click
  **Stop rollout & drag**. This ends the rollout without homing, waits for
  runtime cleanup to release the controller, then enters the selected drag.
  **Cancel drag** cancels that pending transition; the rollout still stops.
- Once cleanup finishes, select **Drag arms: A / B / AB** and **Start drag**.
  A is the left arm, B the right. Use **Exit drag** to leave drag mode and
  disable the selected arms. Then **Return Home** homes without creating a
  rollout; it can also be used directly without dragging first.

After a physical emergency stop, release the E-stop before clicking **Start
drag** or **Return Home**. The drag worker uses teleop's `ensure_clear` to clear
latched errors and verify all selected arms before enabling torque mode. If any error remains,
neither arm enters drag. Direct **Return Home** also checks and clears latched
controller errors before enabling position mode or gripper control. It retries
at most five times, 0.3 seconds apart, and requires fresh, error-free feedback
from both arms before proceeding. It only clears arms in `active_arms`; a fault
on another arm blocks homing and is reported without clearing that arm. If
clearing fails, no homing targets are sent and **Last error** shows the remaining
controller state/error. Normal rollout connection does not automatically clear
faults, and an E-stop during homing aborts the move without retrying recovery.
The panel reports the interrupted rollout and keeps
the underlying error in **Last error**, including when startup failed before
an episode could begin. Recovery state is restored from periodic service
heartbeats even if the one-time failure event was lost. A service disconnect
keeps the recovery panel visible but disables motion until it returns.

Recovery requires `robot.options.execute: true`. Drag respects `active_arms`;
homing reuses the configured Tianji driver, home waypoints and speed. New
rollouts and homing remain locked during drag and its cleanup. Closing the last
viewer browser tab or losing the viewer connection requests drag exit. A
one-shot `manimux run` exits its runtime and does not offer idle recovery; use
`serve` for this workflow. Automatic completion still follows `home_on_close`.

With `gripper_control: true`, every home operation first completes the joint
move and waits until both arms are within 0.5 degrees of home, then opens both
grippers to at least 0.98 normalized aperture while holding the home joints.
Opening uses the existing force-limited targets and must finish before home
returns or cleanup disables the motors. Joint settling and gripper opening
each have a 5-second timeout; faults or timeouts stop the operation and report
an error. With gripper control disabled, homing only moves the arms.

The drag worker runs in `project/teleop/.venv`, using teleop's `ArmDriver`,
`RobotConnection` and `load_tool_config` with **`--tool umi`**. It matches
`set-state <A|B> drag --tool umi`: joint drag at 250 Hz, K=1, D=0.3,
15 deg/s reference tracking, and separate A/B tool calibration. AB uses one
connection and batched joint commands. It does not start teleoperation sensors
or a policy. It never enters drag with an uncleared controller fault.

By default, teleop is the sibling checkout next to ManiMux. For another layout,
set the path in the **runtime** config (not the model-server config):

```yaml
viewer:
  enabled: true
  robot_adapter: tianji
  tianji_teleop_root: /path/to/project/teleop
```

The worker always uses the runtime's configured robot IP and teleop's current
`configs/tool/umi.yaml`; it does not copy calibration values into ManiMux.
The UI reports startup, active drag, exit and errors separately. Failure to
confirm drag cleanup keeps subsequent motion blocked.

## Measured observation history

`TimestampedCameraSensor` consumes `CameraSubscriber.try_recv_bundle()`, retains
server capture timestamps and does not count repeated polls as new frames. It
maps wall-clock capture time to monotonic time with an offset sampled at startup,
and rejects missing, backward, stale/future timestamps or a local wall-clock jump
above 20ms. The camera server must be on the same machine, or its clock must be
synchronized within the configured state/camera tolerances. These timestamps identify the backend's host receive/callback time, not sensor
exposure time. The camera server obtains each image and timestamp atomically.
This is not a hardware synchronization guarantee.

`HistoryStrategy` uses the existing per-tick `build_submission` plugin hook to
cache measured states. Each new pair of camera frames is matched to the closest
buffered robot state, within 20ms per camera; camera skew is bounded at 40ms. It
selects two distinct measured snapshots around the checkpoint interval, within
40ms, then calls the unchanged standard `manimux` or `rtc` strategy. Warmup and
missing history defer submissions. There is no inference-request-based history,
extra hardware polling thread, or change to control_hz/the main loop.

Serial scheduling and max_chunk_steps remain restricted to the built-in
`manimux` runtime name, so this history plugin does not use them. Process action
decoding checks the constructed strategy instead, which this wrapper delegates:
either its `manimux` or `rtc` delegate may use `policy.action_decoding: process`.
The RTC template selects process decoding; the ordinary template retains inline
decoding unless explicitly selected. The wrapper revalidates the delegated
strategy's full configuration, including RTC delay/horizon constraints. Runtime
construction preserves the wrapper's observation and condition-alignment hooks.

## Action conversion and execution differences

The model freezes each arm's latest measured TCP reference, processes two RGB
frames, predicts relative row-rot6d poses, and returns absolute per-arm base TCP
poses and raw apertures. ManiMux converts through the configured `umi_follower`
tool and configured Tianji IK. The default `ik_backend: analytic` uses the SDK;
optional `diff` uses the ported OSQP velocity solver. See
[differential IK configuration and validation](tianji-diff-ik.md) for binding it
to the shared motion profile. Arm A/robot0 is left; arm B/robot1 is right.

The first action origin includes the checkpoint's first offset. Already expired
source rows are removed before IK and recorded in `source_offset_steps`. Each
remaining knot's SE(3) segment is checked through the selected IK in at-most-4ms
substeps. Analytic retains the original branch/limits/FK checks; diff uses its
rate, position/interference and tracking-lag checks, where `lag_policy: report`
records the lag in `diff_ik_lag` instead of rejecting. The first knot uses
its remaining time to the target; subsequent knots use action_dt. Any invalid
pose, aperture or IK rejects the entire chunk. Intermediate IK samples are
validation points; the shared executor interpolates final **joint** knots.
CalibWrist retains all control-rate TCP/IK samples as dense joint commands and
can precompute them; its async runtime sends them from a separate thread. This
adapter retains model knots, so that dense-command behavior is not reproduced.

For RTC, the history wrapper resamples the actually committed joint timeline at
the new observation's `first_offset + j*dt` targets, masks rows beyond the valid
committed plan, and the embodiment converts weighted conditions through FK. The
model rebases these absolute poses into the new TCP reference and guides the real
DDIM sampler with PiGDM or soft inpainting. No joint-space tensor is passed to
UMI's 20D normalizer. RTC preserves the original source horizon and consumed
indices across both adapter and commit-time trimming. Conditions with no valid
committed overlap are explicitly unconditioned and retain the normal blend.
Guidance, transport and process decoding must fit the deployed timing budget;
a finite offline result alone does not prove that budget is met.

Process decoding starts one child per arm and warms its own kinematics/solver
before robot connection. Both receive the same measured decode-time seed and
source clock. Inference and decoding occupy one in-flight slot; the old plan
continues until both valid partitions can be committed together. A failed arm
rejects the whole chunk. A decoder deadline failure faults and cleans up the
runtime. Independent partial-arm holds are unsupported for RTC.

Pause and homing discard the timeline and reset RTC/history state. In-flight
results are drained and rejected using their original request sequence; the
first request after resume uses fresh history and has no old-plan condition.
The measured IK seed can age while the robot follows the old plan, so validate
takeover tracking with the configured executor and motion limits on hardware.

## RTC deployment

Bind the actual checkpoint and explicitly select the existing IK backend:

```bash
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --checkpoint /path/to/trusted/pass_ball.ckpt \
  --runtime-template configs/umi_dp/tianji/infra/pass_ball/rtc.yaml \
  --ik-backend diff \
  --bind-runtime-config data/experiments/pass-ball-rtc.yaml
```

For a station with a reviewed custom profile or diff-IK options, create a new
runtime template from that station's bound configuration. Set
`policy.options.history_strategy: rtc`, `policy.action_decoding: process`, and
keep `execution.runtime: manimux.integrations.umi_dp_tianji.history:build_strategy`.
Remove `execution.inference_schedule` and `execution.refill_threshold_s`; RTC
rejects these unused fields. Preserve the selected profile, IK limits, gripper
semantics and executor settings. Rebind the copied template to produce a new
paired configuration; do not change artifact identities by hand.

`execution.rtc.initial_delay_steps` is an initial estimate in **model action
steps**, not 250 Hz ticks. The template uses 4, `min_execute_steps: null` (half
the source horizon, bounded by feasibility) and PiGDM `beta: 5.0`. Runtime
forecasting takes the maximum of the recent delay buffer and rounds fractional
steps upward. It includes observation age, request preparation, transport,
model inference, both decoders and commit lead. Check `rtc_delay_ms`,
`measured_delay`, `forecast_delay` and `rtc_delay_infeasible`; the feasibility
window is `d <= executed <= H-d`, requiring `2*d <= H`. Calibrate the initial
estimate against the deployed latency distribution. At 30 Hz, 7 steps cover
approximately 233 ms. The first chunk is unconditioned.

Both pass-ball infra templates enable the shared executor's `close_latch` mode:

```yaml
execution:
  smooth:
    gripper:
      mode: close_latch
      close_threshold: 0.6
      open_threshold: 0.75
      closed_value: 0.2
```

Each arm starts logically unlatched. A reference aperture strictly below .6
latches a .2 close target; while latched, values below .75 keep that target.
A value at least .75 releases the latch and follows the reference aperture,
without snapping to a fixed open value. For example, `.60, .59, .74, .75`
produces targets `.60, .20, .20, .75`. The optional `min_closed_s` and
`open_confirm_s` both default to zero, so reopening has no additional delay.
`closed_value` is a target, not a mechanical lower bound: starting below .2
still approaches the target under the configured motion limits.

Latch state advances only on the current executed reference, independently per
arm. Future horizon rows, discarded candidates and held-arm references cannot
change it. Plan replacement and inference gaps preserve the latch; gaps also
hold the last gripper command instead of replacing it with contact feedback.
Explicit Pause, Home and executor reset clear the logical latch. Active control
ticks record `gripper_decision` events with `latched_closed`, `desired_aperture`,
`target_aperture` and `command_aperture`.

The threshold rule matches CalibWrist's current close latch. Its placement
differs: ManiMux applies it to the current timeline sample after interpolation
and blending, whereas CalibWrist maps action knots before interpolation. Exact
transition timing and command-trajectory parity are therefore not claimed.
The model and stored policy actions preserve raw aperture predictions. Shared
motion limits, driver grip margin and runtime timing remain separate settings.
To restore continuous targets, set `mode: continuous` and `closed_value: 0.0`;
continuous mode uses that value as its lower aperture bound.

## Validation

```bash
.venv/bin/python -m pytest tests/unit/test_umi_dp_tianji.py -q
.venv/bin/python -m pytest tests/unit/test_executors.py tests/unit/test_config.py \
  tests/integration/test_mock_run.py -k 'close_latch or umi_pass_ball' -q
.venv/bin/python -m pytest tests/unit/test_tianji_rtc.py \
  tests/unit/test_rtc_runtime.py tests/integration/test_tianji_rtc_process.py -q
bash -n XPolicyLab/policy/UMI_DP/*.sh
python -m compileall -q XPolicyLab/policy/UMI_DP
```

A real libKine decode probe is also available:

```bash
.venv/bin/python scripts/validation/umi_dp_tianji_decode_probe.py
```

On the integration machine, reachable synthetic two-arm chunks took about
19–23ms for H16 and 77–79ms for H64 (three trials, no hardware connection).
These are analytic-backend timings. The diff backend took 65–66ms for H16 and
260–262ms for H64; its individual QP steps had medians of 0.22–0.26ms.
Whole-chunk work exceeds the 4ms budget of a 250Hz tick. Inline decoding blocks
the control loop until it finishes: the 2026-09-14 H64 diff rollout froze for
111–137ms at each of its 18 plan commits, with no commands sent meanwhile.

`policy.action_decoding: process` (manimux or rtc delegate) moves decoding into
one spawned process per arm. Each child builds its own kinematics and QP solver
and warms them up before robot connection. The control loop keeps executing the
previous plan and commits the new one after both arms finish. Offline, per-arm
diff H64 decodes of that rollout took 53ms (left) and 58ms (right) median, versus
115ms serially; on hardware, submit to result took 64ms median and 76ms maximum.

Seeding IK from the measured state at submission failed on hardware: both
2026-09-14 process rollouts were stopped by the 5-degree tracking guard about
125ms after a plan commit. In 17 of 17 moving commits the new plan's first row
lay behind the arm (median 1.97 degrees), so the command reversed at up to
~40 deg/s without an acceleration limit. `pass-ball-h64.yaml` therefore sets
`execution.expected_decode_s: 0.065`, seeding from the previous plan's reference
at the expected commit (see [process decoding](parallel-ik-execution.md)). On
those recordings that seed lay ahead of the arm in 16 of 17 commits (median
2.29 degrees); this replay does not rerun IK, and the fix is unverified on
hardware. A failure
in either arm still rejects the whole chunk, so the previous plan continues; a
decoder exceeding the request deadline still faults the runtime. Process
isolation alone does not establish a hard real-time bound; real-robot command
timing, tracking and task success still require hardware validation, including
for the RTC template, which also uses process decoding. These measurements do
not justify relaxing IK checks or lowering the unified control frequency.

See the UMI_DP README for real recorded-window forward/parity and shared-server
commands. Runtime tests cover capture identity, wrong-time history rejection,
clock jumps, RTC first offsets and tail masks, FK mapping and whole-chunk
rejection. Model training and camera/CAN motion have not been exercised by this
integration. Preserve real checkpoint and hardware timing results separately
from these hardware-free contract tests.

Recorded evidence: [2026-09-13 validation report](umi_dp-tianji-validation.md).
