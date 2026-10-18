<a id="runtime-robogui-and-configuration-ownership"></a>

# Runtime, RoboGUI and configuration ownership

Source paths below are relative to the repository root.

## Experiment lists and idle selection

`manimux/experiment_list.py` loads a finite catalog through the existing CLI loader.
`serve --config-list` accepts entries with unique IDs, config paths relative to the
list, optional display labels and an optional `policy` group (default: `default`).
Task/method/rubric still belong to each resolved experiment. The first entry is selected
initially. Entries in one policy group must share model initialization, policy I/O and
sensor contracts. Across groups, policy clients, adapters, horizons, observation readers
and model identities may differ. All entries must share the complete robot config,
camera-server recipe and RoboGUI robot; output directories must differ. Different model
services must use distinct client endpoints in externally managed mode. Method labels
are unique per policy/task.

For multiple groups, RoboGUI shows Policy and a Switch policy button. Changing the
dropdown stages the choice and disables Prepare until the operator applies it or restores
the active policy. The button selects a complete experiment, preferring the active task
and method label when available. It uses the existing acknowledged idle-selection path;
the policy display group never overrides an experiment's model identity or endpoint.

Lists and entries may also reference an `initial_poses` YAML with named `poses`.
Each preset adds a choice alongside the unchanged default, preserving the policy group,
model identity and service recipe. Presets require configured start motion and provide
finite joint coordinates for every group plus a positive duration. Expanded choices
record pose identity, source and a digest, and use `initial_pose/<id>` output subdirectories.
Changing a pose after collecting sessions requires a new ID. RoboGUI keeps method labels
unique, filters pose choices by policy/task/method, and uses the acknowledged idle-selection
path; selection alone never moves hardware. Prepare uses the selected start pose.
Catalogs without presets keep their existing choices and hide the pose dropdown.

`RuntimeSessionService` accepts selection requests only in its idle polling loop,
after evaluation and recovery. Requests carry the current run_dir; stale Prepare
requests also have to match the selected ID. The GUI retains a selection request
until acknowledgement, disables Prepare while pending, and refreshes task, rubric,
reference layout and progress together. Each selection creates a separate session
manifest with list/selection provenance and a frozen resolved config. Each Prepare
constructs a fresh runtime using the existing worker reset/warmup lifecycle.

Optional top-level `policy_services` maps every policy group to `command` (an argv list),
`cwd` (relative to the list), `env` (string values; null removes an inherited variable),
`startup_timeout_s` (180 by default) and `stop_timeout_s` (30). `{config}` and `{local}`
arguments expand to the selected experiment/station paths. Use the intended model Python
and remove the hardware SDK's PYTHONPATH when it targets a different Python ABI.

`manimux/policy_service.py` owns a pollable process lifecycle, not model implementations.
The initial selection starts when serve begins. Changing policy terminates the owned
process group, waits for exit (SIGKILL after the stop timeout), then launches the selected
command without a shell. Changing methods in the same group reuses the model. TCP
listening on the explicit loopback client endpoint admits Prepare; this is not a model
identity check. Startup exit/timeout disables Prepare and exposes an error/log path.
Retry policy creates another selection/session and retries; another group can also be
selected. Serve shutdown stops its owned model. External port occupants are never adopted
or killed. Managed groups can reuse an endpoint because only one model is kept alive.
The shared camera service is external and unaffected. Logs go to `policy-server.log`
under the session that launched that model.

Without `policy_services`, model servers remain externally managed and switching only
selects a client. In both modes, the next Prepare connects the selected worker and validates
identity/capabilities before connecting the robot; an unavailable or wrong backend fails
that attempt and the idle service allows another selection. Multi-task recipes keep deployment
identity fixed and supply the actual prompt via run.task in each observation.
Changing model initialization or YAML on disk requires restarting the affected
service. Public operation and paths: [Task & method switching](../../../../guideline.html#experiment-lists).

## Inference algorithm versus executor

Current experiment entries explicitly select `executor.type: direct` for every
inference method. New experiment configs follow that selection; Smooth/MPC remain
available implementations, not the current experiment baseline.

Ordinary asynchronous chunking is named **Async** (`inference.algorithm: async`).
Serial selects `inference.algorithm: serial`; its schedule defaults to `serial`.
Both use the same chunk strategy implementation with unchanged timing behavior.
ManiMux remains the framework/package/CLI name. `manimux` is not an algorithm or
strategy selector; configs must select `async` or `serial` explicitly. Runtime
metadata uses these names. Existing record metadata and output directories are
not rewritten. Strategy plugins for ordinary asynchronous chunking report `async`.
Inference preset and experiment filenames use `_async` for this algorithm, and
RoboGUI labels use `Async`. Offline readers may accept `manimux` only when reading
historical recording manifests; it is not a supported live configuration selector.

Read `manimux/runtime/scheduling.py` (`RequestScheduler`) and
`manimux/runtime/inference.py` (`InferenceStrategy`, `build_inference_strategy`).
The scheduler owns request admission: in-flight capacity, optional frequency cap,
ordinary refill/continuous triggers, and serial execution completion. It calls the
strategy only when transport admission allows a request. The strategy retains
algorithm readiness (RTC/PAINT execution windows, ACT query interval, selected
AAC/AutoHorizon prefix), sampling fields, chunk preparation and response feedback.
Do not reimplement transport busy checks in each algorithm.

Ordinary Async can opt into `request_trigger: progress` with `single_inflight`.
The shared `chunk_policy_steps` becomes its source-progress threshold rather than
`max_chunk_policy_steps`; retain the full model horizon. `DefaultChunkStrategy`
updates its source offset only on accepted plans and adds `timeline.cursor(now)`
to count latency-trimmed rows like RTC/PAINT. Reset clears accepted progress;
warmup and rejected responses do not advance it. The scheduler still owns capacity,
and this mode does not consult `refill_threshold_s`. Validate K < H, no source cap,
`drop_infer_latency`, and no waypoint or skipped-row handoff.

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
See [scheduling and action clocks](../../manimux-inference/references/inference.md) for the config matrix.

The optional `StrategyActionDecoder` protocol declares `decode_action(raw, context, *, adapter)` and decodes
candidate predictions using the existing adapter. `decode_strategy_action` uses it
for inline execution and warmup; other strategies retain the original single decode.
The optional `StrategyChunkSelector` protocol declares `select_chunk(chunk=..., now_ns=...)` and defers candidate selection
until immediately before commit. For this path, selection and `prepare_chunk` use
the same timestamp as timeline commit, and recording captures the selected full
prediction before prefix preparation. Warmup selects only a preview and never
accepts a plan. Strategies without this hook retain their existing preparation path.

BID uses these hooks with adapters declaring `supports_candidate_decoding`; it retains
its full reference and timing only after plan acceptance. Its `hold_last_step`
flag gives each executed prefix row a full policy interval. The timeline's shared
`first_future_step` defines latency rounding for both selection and commit.
Model sampling and wire conversion remain in the backend.

`EnsembleSource.ensemble_scope` declares the allowed contribution range. With
`full_prediction`, the wrapper fuses/stores the complete prediction and then calls
the source's `prepare_chunk` to crop execution. With `execution_prefix`, it calls
`prepare_chunk` first and fuses/stores only its output. Preparation runs once;
RTC/PAINT use it for validity checks and currently return the full horizon. Fixed
BID and Serial ACT use full_prediction; AAC, AutoHorizon and BID+AAC use
execution_prefix. The wrapper must not dispatch on algorithm names. Rejection
removes pending candidates and rejected ensemble contributions; reset clears both
the wrapped strategy and ensemble history. Warmup may decode/select previews but
never advances accepted plan history.

For user-facing enable/disable examples and the distinction between RTC sampling,
Timeline seam blending and output ensembling, use
[Supported combinations](../../../../guideline.html#optional-algorithm-composition).
The combination flag defaults off; standalone ACT remains enabled by its algorithm
selection regardless of that flag. Do not present a valid combination as a measured
improvement, or change the contribution scope as a cosmetic refactor.

`RequestState.expected_decode_ns` carries the shared `DecodeForecast` value at
submission (zero for inline decoding). Strategies consume this snapshot instead
of maintaining another decode window. Train-time RTC's `decode_max` estimator
combines it with the last accepted request's submission-to-worker-completion time
and current observation age, counting prefix rows from the action origin; see
[the timing contract](../../manimux-inference/references/inference.md#train-time-rtc). The current model
latency is not yet observable when a hard prefix must be submitted.

`manimux/runtime/timeline.py` owns `ActionTimeline`: committing and sampling decoded
joint trajectories on the runtime clock. Keep chunk handoff here and in the
strategy. Do not fake measured state or modify observations to implement another
algorithm's overlap/truncation behavior.

Serial scheduling uses `algorithm: serial`, `inference_schedule: serial` and
`action_start_mode: first_step_when_ready`. It accepts inline or process decoding;
the existing in-flight flag covers both inference and decoding. A decoded chunk
must retain source offset zero. `ActionTimeline` limits dense runtime rows to the
selected source prefix before shifting execution to commit time. The next request
waits for that execution window to finish; physical measured arrival is separate.

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

`handoff_skip_steps: S` is an execution-only prefix crop in `ActionTimeline.commit`,
after inference, waypoint planning and diff IK have finished. With the same request
and outgoing reference, changing `S` must not change RTC conditions or weights,
model output processing, IK seeds, waypoint targets, or the decoded joint path.
UMI history alignment retains its trained first-action offset and never shifts
conditions to compensate for a future crop.

The timeline first verifies the uncut waypoint's reference and handoff clock, then
removes `S` source rows and the corresponding duration of dense runtime samples.
The start times stay fixed; later decoded points occupy the earlier time slots,
and the end moves earlier by the skipped duration. No replacement lead-in is
generated. A nonzero crop can remove the continuous waypoint seed and introduce
a position jump; the executor still applies its configured limits. Waypoint chunks
require `blend_policy_steps: 0`. Missing/late/changed waypoint validation remains
independent of skip; there is no shifted-condition rejection.

The first chunk is unskipped. Choose `S < H`; a crop that exhausts the usable decoded
suffix is rejected. RTC keeps the original `H` decoded rows and accounts for the
actual committed trim when reading its cursor. The next request therefore sees
the surviving outgoing actions, without offsetting its new condition by `S`.
An empty overlap uses default sampling. Numeric initial delay is validated against
the model horizon (`2*initial_delay_policy_steps <= H`), not a skip-shifted mask.
Measured delay remains wall-clock latency and excludes intentional skip.
Acceptance events distinguish `rtc_source_horizon` (`H`), `rtc_effective_horizon`
(`H-S`, before latency trimming), and `rtc_handoff_skip_steps`.

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

Idle recovery between rollouts (**Clear error**, **Return Home**, **Drag**, **Release**) is
scheduled by the `manimux serve` session and implemented by the robot assembly. The
session reads the assembly's `recovery_actions`, `hand_guiding_actions` and `drag_selections`.
It calls `clear_errors()`, `recover_home(stop)` or the selected hand-guiding handler on a
fresh, disconnected assembly using the configured construction options. Hand-guiding
handlers receive groups, a stop event and an active callback; the robot owns mode entry,
duration, monitoring and cleanup. Loss of the recovery lease cancels these handlers. Service
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

`records.py` reads finalized experiment trajectories; `action_replay.py` owns
hardware-free display and its playback loop. The independent `python -m manimux.robogui.replay`
entry point selects an episode and stream. Dashboard has no historical replay panel.
Shared offline preset/model loaders live in `robot_view.py`. Keep these consumers out of the control loop.
The [research workflow](../../../../guideline.html#experiments) describes current UI and identity fields.


### Prepared layout and explicit warmup

Prepare snapshots the study rules and trial fields. Until first Start, RoboGUI permits
layout/reference changes. Start sends only `layout_id` and `reference_layout` with the
active episode ID and run directory. The runtime revalidates them against the prepared
template, updates episode metadata, and freezes the identity before execution. Subsequent
Pause/Resume cannot relabel the trial. The reference image ID supplies the layout ID when
an image is attached; its hash is recomputed at Start.

With `run.warmup_before_start: true`, a GUI-controlled rollout starts in `idle` and requires
**Activate Warmup**. Activation stays paused and submits display-only requests. Start
waits for the in-flight request, then acknowledged backend reset; only a fresh subsequent
observation enters formal execution. Warmup timings remain separate from formal metrics.
The control mailbox suppresses activation/selection on stale transport snapshots.

Warmup supports inline and process action decoding. Process warmup reuses the
runtime-owned decoder while formal scheduling is paused, keeps the inference request
pending through decode completion, and drains both stages before backend reset. Its
previews use the request observation as the IK seed and never enter the Timeline.
History strategies collect measured observations on every warmup tick and may return
no submission until a valid window is available; UMI DP supplies its measured two-frame
window to the delegate's warmup builder. Restart `serve` after changing a catalog's
warmup settings; the catalog is loaded once at service startup.

PAINT implements `build_warmup_submission()` for ordinary and anchored-prefix requests.
Its `warmup_required_calls` requests two successful calls per branch: the first compiles
that branch, the second supplies a steady end-to-end sample. An early Start remains in
the draining phase while these calls finish, then waits for reset acknowledgement.
Failed required calls cannot release formal execution. Metadata reports required and
completed branch counts and `branches_ready`; warmup plans never enter the Timeline.
The delay buffer uses the selected algorithm's window. PAINT's `seed_warmup()` preserves
ceil-rounded steady samples across reset, with its explicit initial delay as a floor;
first calls of both branches remain visible but are excluded from calibration.

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
XPolicyLab. See [the composition timing contract](../../manimux-inference/references/inference.md#optional-algorithm-composition)
for reference history, horizon validity and unsupported combinations.


## Command-continuous handoff and RTC command prediction

`inference.blend_mode: command_cubic` explicitly selects a Hermite bridge from the
last issued joint command and nominal-control-tick velocity at commit to source row
`trim + blend_policy_steps - 1` and its outgoing tangent. The source row grid and
RTC execution cursor are unchanged. The plan activates at commit, before its first
retained source row if necessary. Sampling and recorded committed rows include the
bridge; canonical raw rows remain untouched. Gripper indices use a monotone smoothstep.
`linear` remains the default. Cubic requires standalone Async/RTC with Smooth, complete-group decoding and at least
two blend rows; it rejects adapter runtime trajectories, waypoint handoffs and serial holds.

RTC `condition_source` selects `canonical` (default), `timeline`, or `executor`.
The optional strategy `set_condition_predictor(callable)` hook receives the executor's
`predict_commands(timeline, origin_ns, dt_ns, steps, group_order, now_ns)` callable.
It returns a valid-row mask and packed absolute joint rows without mutating live state.
Smooth predicts by copying its filter state and stepping the accepted timeline at
nominal control cadence. Its recent command history covers observation timestamps
before the latest command. Reset/hold clear this history. Missing delay-prefix history
falls back to unconditioned inference; missing future rows get zero weight. Feedback
pose guards and latched/discrete grippers are unsupported. This predicts commands,
not measured motion, and does not anticipate a replacement plan or control jitter.

`ActionContext.decode_reference` carries a separately timestamped accepted joint
horizon to inline and process adapters. SAPolicy's opt-in
`policy.adapter.ik_seed_source: previous_plan` interpolates the first decoded row's
seed from that horizon, falling back to the existing runtime seed outside coverage.
Later IK rows retain sequential seeding. Measured observations are not modified.
This reduces seed-time mismatch; it does not enforce branch continuity or cap steps.


## Chunk IK and execution-time IK

`execution.ik.mode` is an experiment setting: `chunk` (default) retains adapter
chunk decoding, while `step` keeps an absolute TCP trajectory until each control
tick. `execution.ik.budget_ms` is optional and defaults to half the control period.
It must be positive and less than one control period.

The step path is adapter pose decoding -> pose ActionTimeline -> one sampled
pose per group -> ExecutionPipeline (optional PoseSpeedLimiter -> StepIKResolver
-> DirectExecutor) -> existing safety/send path.
A pose row is `[x, y, z, qw, qx, qy, qz, gripper]`, in each robot group's base frame,
with the configured TCP and normalized gripper. Decode relative outputs against
the request observation anchor, never against a later execution seed. Timeline
interpolation and blending use linear xyz/gripper and shortest-arc quaternion
SLERP. Record `absolute_tcp_pose` plan chunks; scheduled/command rows remain joints.
RoboGUI draws pose-plan positions directly, without FK on pose rows.

The execution stages are built once and IK is warmed before the control loop. IK starts from
the current control tick's measured joint state and solves only the current target.
The pipeline receives the last sent command separately for failure holds and
optional target limiting; the IK solver does not choose fallback commands. A shared budget
covers all arms. If any arm fails, is nonfinite, or returns late, hold all groups
at the last command and reset executor velocity history. `step_ik_hold` records
failures; `run.control_timing: true` records the `step_ik` stage. YAM checks a
cooperative deadline between native solver iterations. Other native solvers are
checked after return; a long native call cannot be preempted. This is a soft
budget, not a hard real-time guarantee or proof of physical tracking.

Step mode requires direct execution, inline pose decoding, atomic groups, and
linear blend handoff. Smooth/MPC future-joint references, joint candidate scoring,
temporal ensembling, command-cubic blending, waypoint handoff and decoder diffIK
substeps are rejected explicitly. `execution.ik.backend` selects `analytic` (the
existing native solver, default) or `diff` (the shared Tianji velocity QP).
Keep adapter `ik_backend` omitted or `analytic` for pose-only decoding; execution
backend selection is independent of the adapter's chunk solver.

For step + diffIK, set `execution.ik.backend: diff` and optional
`execution.ik.diff_ik` tuning using `DifferentialIKConfig` fields, except
`max_velocity_rad_s`: the rate bound comes only from
`executor.motion_limits.arm.max_velocity`. It requires composed Tianji kinematics
with seven arm joints and one tool coordinate. One QP update uses the measured
joints and nominal control period, with the composed tool offset removed once.
`lag_policy: report` permits bounded progress with residual error; `abort` (the
solver default) rejects excessive lag. This is rate control, not exact pose IK.
QP state resets after rejection, pause, Home, timeline gaps and episode warmup.
Successful updates record `step_diff_ik` events (including lag after optional
pose limiting); failures retain atomic `step_ik_hold`. The direct executor and
command safety checks still apply. No trajectory time stretching is introduced.
Ordinary joint adapters need no IK mode and reject `step`.

Serial/async support all pose-producing IK adapter families. RTC/PAINT require an
explicit pose condition codec: OpenWAM YAM and XR1 YAM support both; Pi05 Tianji,
XR1 Tianji and UMI-DP Tianji support RTC. The other adapters reject conditioned
step modes. RTC executor forecasts are unsupported. `InferenceRequest` carries
`action_condition_space` explicitly, since Tianji joint and dual-pose arrays can
both have width 16. A pose condition goes directly back to model coordinates;
conditioning does not solve future IK.

Example: `configs/experiments/put_bottles/openwam/yam_openwam_serial_step30000_step_ik.yaml`.
Existing presets retain chunk IK; the new example is not activated automatically.


### Optional pose target speed limit

`execution.ik.pose_speed_limit` defaults to `null` (off), including in step mode.
An enabled mapping accepts `max_linear_m_s` and `max_angular_rad_s`; provide at
least one positive finite number, leaving the other absent/null if unrestricted.
This option requires step mode. It does not modify legacy chunk IK.

The independent `PoseSpeedLimiter` limits the current sampled TCP target before IK. Its reference is
FK of the last actually sent joint command, while the IK initial configuration
is the measured state. One interpolation fraction bounds translation and shortest
rotation by speed times the nominal control period. A stalled tick does not earn
a larger catch-up step. The gripper remains governed by its existing executor
limits. The limiter keeps no independent cursor: failed IK and executor limiting
cannot advance a hidden target. Pose FK and limiting are skipped when disabled.

This is a target-increment limit, not a guarantee on measured velocity or the
joint path between targets. IK tolerances and downstream joint limiting still
apply. It does not stretch Timeline timestamps, rewrite RTC/PAINT conditions, or
wait for limited targets to be reached. Under sustained limiting, execution may
lag and intermediate source targets may be skipped; evaluate that tracking error
before using it in a timing-sensitive rollout. This intentionally differs from
ManiUniCon's waypoint-time stretching. `ExecutionResult.ik_status` includes
`pose_speed_limited` fractions for the affected groups; hold events include them.

```yaml
execution:
  ik:
    mode: step
    pose_speed_limit:  # omit this mapping, or set null, to disable
      max_linear_m_s: 0.25
      max_angular_rad_s: 0.6
```


### Execution component responsibilities

`runtime/execution.py:ExecutionPipeline.step` is the active-tick execution entry.
It does not receive model responses, accept chunks, change Timeline timestamps,
perform safety validation, or send hardware commands. The control loop retains
those orchestration/lifecycle duties and records returned execution diagnostics.

- `runtime/pose_limits.py:PoseSpeedLimiter` clips one TCP target per group. It owns
  no IK, executor, trajectory clock, or independent progress cursor. It is absent
  when limiting is disabled.
- `runtime/ik.py:StepIKResolver` solves that target from measured joints. It returns
  a joint reference on complete success, or `None` if any group fails. It does not
  limit target speed, select a hold command, reset an executor, or send commands.
- `ExecutionPipeline` composes the optional stages and the existing joint executor.
  It includes pose limiting and all arm solves in one deadline. On IK failure it
  resets executor velocity history and returns an atomic last-command hold.
- The existing `Executor` receives only joint references, including its unchanged
  full reference horizon in chunk/joint mode. It owns joint command generation.

The pipeline returns `ExecutionResult(reference, command, ik_status)` so scheduled
joint recording and failure events remain visible without reproducing execution
logic in the control loop. The same executor instance remains available for
existing pause, gap hold, reset and forecast behavior. No waypoint scheduling or
waypoint-time stretching is introduced by this component split.


### Tianji step-IK integration (2026-10-14)

`experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999_serial_step_ik.yaml`
is an offline-first Serial 32 example: 100 Hz control, 30 Hz policy, inline pose
decoding, direct execution, native `analytic` IK, a shared 5 ms budget and no blend.
TCP target limits are 0.4 m/s and 1.2 rad/s. Existing experiments retain chunk mode;
do not change only `execution.ik.mode` on process/diffIK/waypoint configurations.
Remove adapter diffIK/substep/waypoint settings, select inline/direct, and express
TCP limits under `execution.ik.pose_speed_limit`. These are per-tick target limits,
not the old decode-time waypoint path or a promise of identical joint motion.

SAPolicy step IK is rejected until its adapter-owned TCP geometry and bounded
solver configuration can be shared with execution. A custom virtual TCP cannot
be labeled as the robot's configured TCP. Chunk-mode SAPolicy remains supported.
Startup warms the bounded native solver when available, with a separate startup
allowance; actual ticks still share their configured budget across both arms.

Execution evaluation accepts joint and TCP committed plans. Pose-plan seam and
window metrics use Cartesian position directly; interpolation uses shortest-arc
SLERP. Smoothness continues to use the recorded post-IK scheduled joint reference.
An expired pose plan requires observed constant joint commands to qualify a hold;
its pose endpoint must never be compared to a joint vector. These reference
metrics do not measure actual physical tracking accuracy.
