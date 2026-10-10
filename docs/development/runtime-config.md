<a id="runtime-robogui-and-configuration-ownership"></a>

# Runtime, RoboGUI and configuration ownership

Source paths below are relative to the repository root.

## Inference algorithm versus executor

Read `manimux/runtime/scheduling.py` (`RequestScheduler`) and
`manimux/runtime/inference.py` (`InferenceStrategy`, `build_inference_strategy`).
The scheduler owns request admission: in-flight capacity, optional frequency cap,
ordinary refill/continuous triggers, and serial execution completion. It calls the
strategy only when transport admission allows a request. The strategy retains
algorithm readiness (RTC/PAINT execution windows, ACT query interval, selected
AAC/AutoHorizon prefix), sampling fields, chunk preparation and response feedback.
Do not reimplement transport busy checks in each algorithm.

On each eligible control tick, the scheduler calls the optional
`observe_snapshot(snapshot)` hook **before** admission checks. DP/UMI history plugins
use it to retain camera/state samples while inference is busy. `build_submission`
then consumes that history and may return `None` for algorithm reasons. Call
`scheduler.on_submitted(now_ns)` only after submitting the prepared request, and
reset the scheduler together with the strategy. Warmup keeps its independent,
reset-fenced single-request lifecycle. Request-sequence invalidation, deadlines,
and paused/Home response filtering remain in the runtime.

`inference.strategy` still selects `module:factory` or a
`manimux.inference_strategies` entry point. Existing history wrappers share the
same scheduler as their delegate. Streaming/continuous custom strategies remain
unsupported until their lifecycle and action-reference handling are validated.
See [scheduling and action clocks](../advanced/inference.md) for the config matrix.

An optional strategy `decode_action(raw, context, *, adapter)` hook decodes
candidate predictions using the existing adapter. `decode_strategy_action` uses it
for inline execution and warmup; other strategies retain the original single decode.
An optional `select_chunk(chunk=..., now_ns=...)` hook defers candidate selection
until immediately before commit. For this path, selection and `prepare_chunk` use
the same timestamp as timeline commit, and recording captures the selected full
prediction before prefix preparation. Warmup selects only a preview and never
accepts a plan. Strategies without this hook retain their existing preparation path.

BID uses these hooks with adapters declaring `supports_bid_backward`; it retains
its full reference and timing only after plan acceptance. Its `hold_last_step`
flag gives each executed prefix row a full policy interval. The timeline's shared
`first_future_step` defines latency rounding for both selection and commit.
Model sampling and wire conversion remain in the backend.

`manimux/runtime/timeline.py` owns `ActionTimeline`: committing and sampling decoded
joint trajectories on the runtime clock. Keep chunk handoff here and in the
strategy. Do not fake measured state or modify observations to implement another
algorithm's overlap/truncation behavior.

Read `manimux/runtime/executors/base.py`: an executor implements `reset(state)`,
`horizon_steps` and `step(now_ns, state, reference) -> RobotCommand`. It consumes
`ActionHorizon` and owns command generation/smoothing/configured limits, not model
requests or hardware sessions. Follow the actual executor construction in
`runtime/edge.py` when adding a type; do not assume the policy plugin loader applies.
Per-group gripper indices come from assembly layout, not a universal last-column
assumption. Reject unsupported behavior rather than silently choosing another mode.

`executor.smooth.gripper.mode` is a tagged, mutually exclusive aperture mapping:
`continuous` forwards normalized model openings, `curve` applies its required
`deadzone` and `exponent` before IK and timeline commit, and `close_latch` owns the
stateful close/open thresholds in the executor. Do not place curve parameters in
`policy.adapter` or latch thresholds beside a continuous/curve mode. Common
`group_indices` and motion rates remain valid for every mode. The loader derives the
adapter-side stateless mapping from this single authored block so recorded plans,
IK and hardware commands use the same aperture.

Curve mapping is supported by `JointAdapter`, Tianji absolute-EE adapters (XR-1
and Pi05), and the Tianji UMI-DP adapter. Other adapters reject this mode at
construction. An adapter must declare `supports_gripper_mapping = True` only when
it maps decoded tool coordinates before timeline commit and inversely maps RTC
conditions back into model units. Use the shared `GripperMapping`; leave measured
observations and arm coordinates unchanged. Zero-weight RTC padding stays padding.
`hysteresis` initializes its closed/open state from measured aperture; `close_latch`
retains its separate explicit-reset and inference-hold behavior.

For scheduling changes use a deterministic clock and synthetic chunks to check
timestamps, overlap, delayed/rejected responses and reset as relevant. For executor
changes check actual arm/tool limits and output groups. Preserve capability checks;
an algorithm setting alone does not implement the model's sampling hooks.

### RTC waypoint handoff skip

With `inference.algorithm: rtc`, `handoff: waypoint`, and `blend_policy_steps: 0`,
`handoff_skip_steps: S` uses the same waypoint semantics as ManiMux/UMI-DP: model
row `k+S` occupies source time slot `k`, without moving the handoff time. The adapter
plans the lead-in to that shifted target before IK; the timeline verifies the
reference seed and does not skip the already-shifted chunk a second time.

RTC places the condition for slot `k` and its weight at model row `k+S`. Leading
skipped rows have zero guidance. Requests retain the checkpoint's full model horizon
`H`, but a skipped waypoint plan has only `H-S` effective time slots; discarded
prefix slots are retained for index bookkeeping, not execution. The next request
uses this actual shorter horizon and zero-pads unknown conditioning rows to `H`.
UMI's history wrapper also aligns against shifted slot timestamps while preserving
the checkpoint's first-action offset. No model/server changes are needed.

The first chunk is unskipped. A conditioned response with shifted guidance is
rejected if it cannot acquire its matching waypoint handoff; executing it on the
unshifted clock would violate its guidance. Empty overlap uses default sampling.
Choose `S < H` and, for a numeric initial delay, `2*initial_delay_policy_steps <= H-S`.
The measured delay remains wall-clock latency and does not include intentional skip.
Acceptance events distinguish `rtc_source_horizon` (`H`), `rtc_effective_horizon`
(`H-S` for a skipped waypoint), and `rtc_handoff_skip_steps`.

<a id="robogui-replay-and-recording"></a>

## RoboGUI, replay and recording

The current dashboard loads a RoboGUI YAML through `load_robogui_config()`, then
constructs `RobotView(RobotModel.from_config(model), config)`. Start from
`manimux/robogui/robot_view.py` and `robogui/robots/yam/robogui.yaml`.
A new body normally needs a `robogui/robots/<name>/robogui.yaml` pointing to its
assembled offline model, with camera slots and per-group display styles.
`--robot <name>` selects that directory; `--config <path>` selects an explicit YAML.
Do not create a second kinematic definition just for display.

### Trace `--robot yam` to the rendered model

```text
--robot yam
  -> manimux/robogui/robots/yam/robogui.yaml
     model: ../../../configs/embodiment/robot/yam_dual.yaml
  -> RobotModel.from_config(yam_dual.yaml)
     components.left_yam/right_yam -> configs/embodiment/arm/yam.yaml
     implementation: manimux.embodiments.arm.yam:YamArm
  -> YamArm.load_model(**options) -> ArmModel(geometry, URDF, visual mapping)
  -> RobotView -> each assembled group's visual_urdf()
  -> ViserUrdf -> browser rendering
```

The default YAM URDF is
`manimux/embodiments/arm/yam/assets/i2rt/robot_models/arm/yam/yam.urdf`.
Its path belongs to the component's offline model, not a `yam` branch in the
RoboGUI. Both arms reuse that component; the assembly supplies group identities.
A separate end effector is composed into the group's visual model when present.

During live operation, RoboGUI receives grouped measured positions in runtime
messages. `RobotView.visual_configuration(group, q)` converts the control vector
to URDF coordinates and `ViserUrdf.update_cfg()` updates the mesh pose. For example,
one scalar gripper opening may drive two visual finger joints. This mapping belongs
to the component model; do not change policy action dimensions to match mesh joints.
The RoboGUI provides a visual representation of state and predictions, not a physics
simulation of contact, forces or dynamics. Demo supplies synthetic positions and
replay supplies saved positions through the same display model.

### Add a new body's visual representation

1. Implement the component's offline `load_model()` and supply its URDF/meshes,
   coordinate definitions and visual mapping. Follow [component ownership](components.md).
   Geometry loading must work without constructing an SDK connection.
2. Declare components and groups in the robot assembly YAML. RoboGUI group names and
   incoming runtime group names must agree; use the same geometry definition as adapters.
3. Add `manimux/robogui/robots/<name>/robogui.yaml`, referencing that assembly in `model`.
   Set group styles/initial poses, display placement and camera slots. Copy YAM's YAML
   structure, not its joint counts, physical dimensions or camera assumptions.
4. Load with `--robot <name>`, or keep a custom display YAML elsewhere and use
   `--config <path>`. No new per-body dashboard class or main-loop branch is required.
5. Check loading offline, all mesh paths, group dimensions, joint order, units and
   gripper extremes. Replay a known absolute joint trajectory and verify its rendered
   poses. Then verify measured-state display in an authorized real deployment.

Keep imported asset licenses and mesh references with the component. Do not require
another developer's absolute paths. `robogui_display_frame` arranges the scene; it
must not be used to change TCP offsets or control-space calibration.

The older `robogui/robots/base.py` `RobotAdapter` and its entry-point loader still
serve compatibility consumers such as collection-record replay. They are not the
current live dashboard's integration entry point. A display model is also distinct
from `policy_adapter.PolicyAdapter`, which converts policy inputs/actions.

Keep URDF display mapping distinct from command coordinates: rendering two fingers
does not add two action DOFs. Use offline models/assets for replay; loading an action
trajectory must not open a robot connection. Missing assets should be reported,
not substituted with invented geometry. Replay should not change live inference,
recording or session behavior. Verify action replay with a named NPZ and a finalized recorded trajectory.

Record runtime evidence under `manimux/recording/`. Distinguish predicted actions,
sent commands and measured motion; RoboGUI refresh rate is not control frequency.
Do not reintroduce teleoperation or demonstration collection as part of replay.

Idle recovery between rollouts (**Clear error**, **Return Home**, **Start drag**) is
scheduled by the `manimux serve` session and implemented by the robot assembly. The
session reads the assembly's `recovery_actions` and `drag_selections` and calls
`clear_errors()`, `recover_home(stop)` or `recover_drag()` on a fresh, disconnected assembly
using the configured construction options; it contains no robot-specific code. Service
shutdown sets the recovery stop event before joining the worker. Idle Home implementations
must cancel further motion and release owned resources before returning. For Tianji-TacCap,
Home moves the arms to
`home.joints_deg` on a 9 deg/s cosine joint profile, confirms arrival within 0.5 degrees,
then opens the grippers when end-effector control is enabled. Idle recovery Home enables
gripper control within the Tianji assembly itself. Loading or displaying Home never moves
hardware. The shared `tianji_taccap.yaml` retains the original pass-ball Home; experiments
explicitly select `tianji_taccap_pass_ball.yaml` or `tianji_taccap_pack_plate.yaml` for
their task's destination.

## Which YAML owns the setting?

The composition entry is an experiment under
`manimux/configs/experiments/<task>/<model>/`. Use `manimux.cli.load_config()` and
the existing referenced-config loading; do not create another schema/merge system.

- **Component YAML**, `configs/embodiment/{arm,end_effector,sensor}/`: implementation,
  component model options, action layout and reusable device defaults.
- **Robot assembly**, `configs/embodiment/robot/`: named components/groups, mounts
  and shared body hardware/control configuration. Shared control profiles own
  reusable motion limits; an experiment selects or explicitly overrides them.
- **Private station**, `configs/local/station.yaml` or selected `--local`: physical
  CAN/controller/device bindings and service addresses. Never assume enumeration
  order establishes left/right; do not commit installation-specific bindings.
- **Camera set**, `configs/embodiment/sensor/cameras/`: served source names and
  component selections, referenced by `camera_server.config`. Runtime `sensors`
  selects which sources to read; `policy.adapter.camera_map` maps them to model keys.
- **Deployment recipe**, `configs/policy/<model>/<embodiment>/<task>/`: checkpoint
  and serving/inference settings. `policy_server.config` selects this recipe;
  it does not own robot execution. Model defaults remain in XPolicyLab.
- **Experiment `policy`**: runtime client (`worker`, service/options), action
  interval/horizon/timeouts/delay and inline `adapter` class/mappings. There is no
  required separate YAML for every field. Do not hard-code experiment values such
  as `action_dt_s`, `horizon_policy_steps` or `inference_delay_s` into Python bases.
- **Inference YAML**, `configs/inference/`: algorithm request/chunk rules.
- **Executor YAML**, `configs/executor/`: smoothing and command-generation choices.
- **Experiment run/robogui/recording**: task, output, lifecycle/UI and rollout settings.
  Private training work stays in the ignored root `training/` workspace.

In the list above, `configs/` means `manimux/configs/`.
`policy` describes the runtime side; `policy_server` selects the model-serving side
of the same experiment. Pair their action representation, checkpoint identity and
sampler capabilities. Do not delete identity checks to make a mismatched pair start.

Keep `policy.action_dt_s` (model trajectory spacing), `robot.control_hz` (command
frequency), camera acquisition rate and recording video rate independent. Horizon,
decoded prefix and executed chunk length are also distinct. Do not infer one from
another just because one example happens to use the same value.

## Example and checks

Trace `configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml` under `manimux/`:
robot assembly -> component YAMLs; camera set -> station bindings; policy server
recipe -> client format and joint adapter; RTC -> timeline; direct executor ->
robot commands. Copy only the relevant choices when adding an experiment.

Load the changed YAML through the real loader and inspect its resolved values
without constructing devices or starting services. Check reference paths relative
to their containing file, client/adapter pairing and declared group layout. Do not
treat config loading as SDK, model or hardware validation.

## Minimal display configuration

```yaml
# Place at manimux/robogui/robots/<name>/robogui.yaml.
model: ../../../configs/embodiment/robot/my_robot.yaml
label: My robot
camera_mode: policy
groups:
  arm:
    label: Arm
    initial: [0, 0, 0, 0, 0, 0]
```

This fragment assumes the referenced offline assembly defines a six-coordinate
`arm` group. Use its actual dimension and assets. Verify `load_robot_view()` and
NPZ playback without constructing a driver. Display placement is not calibration.

For executor work, start from `runtime/executors/direct.py` or `smooth.py`, implement
`Executor`, and extend the explicit `EdgeRuntime._build_executor` selector plus the
executor parameter parser in `cli.py`. That is construction-time dispatch, not a
per-tick vendor branch. There is currently no general executor entry-point loader.

The live dashboard's `records.py` reads finalized records; `action_replay.py` owns
hardware-free trajectory display. Keep these consumers out of the control loop.
The [research workflow](../usage/research.md) describes current UI and identity fields.

## Opt-in inference composition

The factory retains one base strategy and optionally wraps it with `EnsembleStrategy`
when `temporal_ensemble.enabled` is true. Composition reuses existing decode, selection,
prepare, acceptance and rejection hooks; it introduces no second runtime loop.
`bid.execution_horizon` is `fixed` by default and accepts `aac` for candidate selection
with an adaptive prefix. These new paths use complete inline absolute-joint decoding.

Serial ACT also uses `EnsembleStrategy`, around a small `SerialACTStrategy` that
reuses default request construction and crops the execution prefix after fusion.
`chunk_policy_steps` sets K (null means H); the asynchronous query-interval alias
is bypassed in serial mode. Full unfused predictions remain in the ensemble history,
anchored at each commit time. The existing serial scheduler and final-row hold enforce
K action intervals; the model horizon, observation timestamps and asynchronous ACT
strategy remain unchanged.

AAC/AutoHorizon retain serial defaults but permit explicit single/multi-inflight
schedules. Asynchronous adaptive preparation caps source rows before Timeline removes
expired rows. AAC defers candidate selection/history until commit for asynchronous or
ensemble paths; plain serial AAC retains its existing client behavior. PAINT streaming
retains the accepted-plan identity with each condition and rejects replaced references.
The streaming wrapper accepts existing PAINT/AutoHorizon sampler modes without changing
XPolicyLab. See [the composition timing contract](../advanced/inference.md#optional-algorithm-composition)
for reference history, horizon validity and unsupported combinations.
