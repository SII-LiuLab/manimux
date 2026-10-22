<a id="runtime-robogui-and-configuration-ownership"></a>

# Runtime, RoboGUI and configuration ownership

Source paths below are relative to the repository root.

Service configuration checks must not import recording storage dependencies.
`manimux.recording.recording_parameters` is lightweight; the package loads
`EpisodeRecorder` lazily only when runtime recording is requested. Verify model
and camera `--check` entry points with `zarr` and `numcodecs` imports blocked.

## Collections, composition and idle selection

`manimux/configuration.py` composes file-relative `_base_` experiment/task references
and resolves named poses. `manimux/cli.py` loads component references and local bindings.
`manimux/collections.py` validates `configs/lists/*.yaml`, assigns stable output identity,
selects trial budgets, checks saved-condition fingerprints and imports matching history.
`manimux/experiment_list.py` is the internal selection/metadata representation.

Formal studies use `run --config-list ... [--experiment-id ...]`. Any concrete
runtime variant can also use `run --config` for interactive unscored trials, or
`run --config --once` for one rollout. `_trial_config` copies the resolved recipe,
clears study history/layout/repeat state and assigns a separate trial output root.
The session and GUI both reject experiment preparation in this route. Camera and
generic policy launchers accept the same two inputs. Config inspection must not
import robot implementations or hardware dependencies. Public `serve` is removed. The stored/internal launch-mode values
remain `serve` (interactive) and `run` (once); these are record/protocol fields, not
public CLI names. Collection recording.root is authoritative; reject
station output overrides and variant output paths. Task/method selection and explicit
ID selection use the same `<root>/<collection>/<task>/<variant>/session-…` output.

`RuntimeSessionService` accepts idle selection after recovery and evaluation. It
checks shared service contracts before creating a session and rejects incompatible
switches. Collections may enumerate incompatible services, but camera/model launchers
require an explicit ID when their recipes differ. No automatic service restart or hot
model reload. Starting-pose-only changes are compatible and take effect at Prepare.

Prepare uses a retained `prepare_request_id`: reading controls does not consume it.
In plain `run --config` demo/trial sessions, RoboGUI exposes an editable **Task prompt**
before Prepare. Prepare freezes the trimmed text in `task_command`; the session copies
it into that rollout's `run.task_prompt` before constructing the runtime, so warmup,
inference and episode metadata agree. Idle heartbeats preserve the local draft, while
a new service seeds it from YAML. Finish before changing instructions. Formal
`--config-list` studies retain configured prompts and evaluation controls; demo sessions
hide layout/scoring controls, including **Edit last evaluation**.

Policy plans retain the frozen task prompt even when the optional policy display label
is empty. Publishing an unlabeled policy must not clear the running task instruction.
Each episode also starts its own first-feedback timeout. An old rollout timestamp must
not mark the next Prepare unavailable or close its warmup preview; episode events do
not count as fresh robot feedback, and a missing first state still expires.

The idle session acknowledges admission with `rollout_prepare_accepted` and repeats
`prepare_ack` in idle heartbeats. The GUI clears transmission only on matching
acknowledgment, episode start, failure or a new service boundary. The session ignores
repeated admission of the same ID; a lost reply must neither strand Preparing nor
create another rollout. Resource cleanup finishes before the session advertises idle.

After Prepare and before first Start, **Back to setup** requests the existing Finish
cleanup. Setup remains locked until the owning runtime has stopped, completed required
Home/release checks and finalized its recording. A GUI Finish with no acknowledged Start
is recorded as `preparation_cancelled`, with the prepared identity retained and
`experiment_mode=false`; it skips evaluation and consumes no repeat. Start is remembered
independently of RPent's idle hold while planning. Once Start is acknowledged, Finish
retains the normal evaluation flow, even if no policy action was dispatched. Reconnects
restore the frozen Start selection so Resume cannot lose its identity.

Finish and active-rollout Home also carry retained request IDs. Reading GUI controls
does not consume them. The runtime control mailbox deduplicates Home by run directory,
episode and request ID; Home completion repeats `home_ack` in state metadata. Shutdown
progress acknowledges Finish. New ID-bearing commands must match the current episode.
`control_revision` travels with Start/Pause and returns in `control_status`: GUI intent
is not runtime confirmation. A transport timeout must not silently discard a requested
Home/Finish. The idle service polls controls every 20 ms by default.

`OperationStatus` reports `runtime_preparing` and `runtime_home` from its own publisher
thread during blocking setup/Home. It reports phase elapsed time and process liveness,
never hardware health. Missing progress heartbeats keep Prepare blocked and show
unknown hardware state; they do not authorize another attempt or release resources.

RoboGUI keeps layout selection per task across algorithm/initial-pose session changes,
falling back only when that layout is unavailable. Rendering and history scans run
outside the control-state lock; asynchronous history results carry a selection epoch.
Reference previews cache decoded pixels by file identity; Start rereads and hashes the
selected image before freezing experiment identity. Camera labels distinguish advancing
frame metadata, stale previews and frozen end-of-trial evidence. Ended-trial images stay
visible until a new trial begins, and recorded camera videos can be downloaded for review.
Evaluation saves run asynchronously with a persistent receipt. Editing the last label
atomically writes `revision` and embeds previous labels in `history`; Skip preserves an
existing label. Run controls retain their position and diagnostic details start collapsed.

`manimux/failures.py` attaches diagnostic source/stage/target at operation boundaries
without changing exception types or recovery decisions. The session retains structured
`failure` metadata and the complete exception chain in both failure events and idle
heartbeats. RoboGUI renders source, reason and next step, with expandable original
details; idle status keeps the category. Missing legacy metadata uses conservative
message classification, and unrecognized failures remain unknown. Never infer DFU,
motor damage or physical readiness from a generic connection error. Camera server
startup/read exceptions include the exact camera name, including before runtime connects.
Robot errors use a transport-neutral title. Explicit native `cartesian_reflex` or
`joint_reflex` errors remain visible through nested cleanup exceptions; classification
does not clear faults or authorize another Prepare.

`runtime/lifecycle.py` publishes `runtime_shutdown` during blocking cleanup. A dedicated
thread owns its ZMQ publisher; it never reads or commands hardware. Phases are stopping,
homing, closing, finalizing and release_blocked, scoped by run_dir and episode_id.
The embodiment's `observe_release` callback reports a manual gate without importing
GUI code. The dashboard blocks controls during cleanup and uses this heartbeat
separately from achieved-state freshness. Reporter liveness does not prove motion
progress or controller health. A lost cleanup heartbeat leaves hardware state unknown.

Both CLI `run --config` and session `run --config-list` call `build_runtime`, which
constructs `EdgeRuntime` for every registered inference strategy. Model workers return
actions and capabilities; they do not own robot Home or release. OpenWAM cache/sampling
variants use the same lifecycle. FastWAM's XPolicyLab adapter alone is not evidence of
a configured ManiMux real-hardware deployment.

Exit capabilities currently differ by embodiment and must not be conflated:

| Assembly | Current exit contract |
| --- | --- |
| YAM | Mandatory `home_before_close`, health-checked Home and measured release gate, independent of model/algorithm. |
| Tianji-TacCap | Has Home, but controller `stop()` disables control; has not opted into YAM's mandatory-Home contract. Audit ordering before enabling it. |
| CAN arm assembly (Piper/ARX) | Home is not implemented; `home_on_close=True` is rejected. Do not advertise automatic Home. |
| Franka FR3 / Robotiq | Home and idle recovery are unsupported. Finish stops the owned tracking session and gripper request, then closes connections; receive-only mode performs no actuator writes. |

Changing one shared path requires a strategy regression matrix and an embodiment
capability audit. The same interface name does not establish equivalent hardware behavior.

A fresh runtime is constructed at Prepare. Manifests retain resolved configuration,
source hashes, collection identity and selection metadata. Historical snapshots are
immutable; matching explicit history inputs contribute to live GUI repeat counting.
The derived inventory is refreshed during operation and by `inspect --index`; labels
remain authoritative. See [the protocol](../../../../guideline.html#experiment-lists).

## Native external actions

`ExternalActionSource` supplies bounded joint rows to `EdgeRuntime` without a policy
worker. The runtime retains the executor, shared `_send_command` validation,
recording and GUI lifecycle. This mode is exclusive with policy dispatch and
uses direct serial execution; every row advances only after successful submission.
Stop is latched until a control-thread hold acknowledgement. The
[legacy native protocol fixture](rpent-native.md) owns environment requests, calibrated path
planning and logical episode receipts without writing hardware.

Startup and cleanup do not assume a policy worker exists. Shutdown closes agent
admission before potentially blocking Home/release operations, then uses the shared
resource cleanup before finalizing recordings. External events and cleanup errors
remain in aborted recordings. `max_control_steps: 0` disables the tick budget;
Finish, faults and external shutdown still terminate the runtime.

## Inference algorithm versus executor

External planners can submit bounded serial policy calls to one rollout through
`manimux-rpent serve`. Its shared mailbox accepts motion and policy calls, holds between calls and receives feedback
after command dispatch, reusing the existing client, adapter and executor. See
[unified RPent tools](rpent.md) for the protocol and lifecycle.

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

Read `manimux/runtime/scheduling.py` (`RequestScheduler`) and
`manimux/runtime/inference.py` (`InferenceStrategy`, `build_inference_strategy`).
The scheduler owns request admission: in-flight capacity, optional frequency cap,
ordinary refill/continuous triggers, and serial execution completion. It calls the
strategy only when transport admission allows a request. The strategy retains
algorithm readiness (RTC/PAINT execution windows, ACT query interval, selected
AAC/AutoHorizon prefix), sampling fields, chunk preparation and response feedback.
Do not reimplement transport busy checks in each algorithm.

For `request_trigger: refill`, `refill_threshold_s` must convert to a finite,
positive integer nanosecond duration using the scheduler's rounding. Reject zero
and positive values that round to zero during configuration validation. Admission
remains strictly below the threshold; an empty or exhausted timeline has zero
remaining duration and is therefore eligible with every valid refill threshold.

Ordinary Async can opt into `request_trigger: progress` with `single_inflight`.
The shared `chunk_policy_steps` becomes its source-progress threshold rather than
`max_chunk_policy_steps`; retain the full model horizon. `DefaultChunkStrategy`
updates its source offset only on accepted plans and adds `timeline.cursor(now)`
to count latency-trimmed rows like RTC/PAINT. Reset clears accepted progress;
warmup and rejected responses do not advance it. The scheduler still owns capacity,
and this mode does not consult `refill_threshold_s`. Validate K < H, no source cap,
`drop_infer_latency` or `first_step_when_ready`, and no waypoint or skipped-row handoff.
With `first_step_when_ready` and an uncropped adapter output, progress starts at
zero on activation, so K policy intervals pass before the next submission.
This is distinct from `min_plan_hold_policy_steps`, which gates replacement,
not submission. Warmup uses the same progress trigger through its private planner.

On each eligible control tick, the scheduler calls the optional
`observe_snapshot(snapshot)` hook **before** admission checks. DP/UMI history plugins
use it to retain camera/state samples while inference is busy. `build_submission`
then consumes that history and may return `None` for algorithm reasons. Call
`scheduler.on_submitted(now_ns)` only after submitting the prepared request, and
reset the scheduler together with the strategy. Warmup keeps its independent,
reset-fenced single-request lifecycle. Request-sequence invalidation, deadlines,
and paused/Home response filtering remain in the runtime.

`inference.min_plan_hold_policy_steps` defaults to zero (direct replacement).
Its primary purpose is to prevent rapid plan replacement during continuous
multi-inflight inference; continuous single-inflight can use it too. It is an
optional replacement gate, not a request cadence, model horizon or control rate.
Ordinary Serial/RTC presets should leave it at zero unless an additional hold is
explicitly needed. Existing nested RTC execution/request parameters are unchanged.
Positive values require resolved single-inflight/multi-inflight and inline decoding.
`EdgeRuntime` retains only the newest eligible raw response until
`ActionTimeline.replacement_ready` permits replacement: count model policy intervals
from the active runtime trajectory's activation, capped by its end. A command-cubic
blend activates at its blend start. Receipt frees the
single-inflight slot; releasing a buffered response must not free a newer request.
Recheck the response's own deadline and accepted sequence at release. Decode,
select/prepare and commit against the release clock, retaining the request's
observation decode seed until then. Supersession, Pause and Home discard pending
state through the strategy rejection lifecycle. RTC/PAINT use their reference-plan
fence for buffered single-inflight too; prefix checks and ACT fusion use aligned
handoff timestamps. The zero path retains its original preparation and scheduling.
This setting does not change RTC's separate request-cadence parameter of the same
name. Usage is in [minimum execution horizon](../../../../guideline.html#minimum-execution-horizon).

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
The configured `max_plan_age_s` still limits observation-to-commit age, including
provider latency; `first_step_when_ready` does not reset that safety clock.
`plan_too_old` events and agent receipts retain the measured age, configured limit,
inference duration and request-scoped response artifact directory. Provider success
and runtime admission are separate outcomes. Deadline, session and action-space
rejections also preserve their actual cause in agent receipts instead of a generic
`policy_response_rejected` message. Do not extend the age limit merely because a
provider response arrives within its separate transport timeout.

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

### Signed Async handoff adjustment

For standalone single-inflight Async using `drop_infer_latency` and blend handoff,
`handoff_skip_steps` may be negative: retain that many latency-trimmed rows at the
same handoff clock, extending the remaining trajectory. Total crop is clamped at
zero and the first chunk is not adjusted. Source-progress triggers count the actual
crop plus the active cursor. This changes timing alignment, not physical speed
limits, and does not guarantee continuous commands. Other algorithms retain their
existing nonnegative restrictions.

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

## Shared command constraints and failed submissions

`EdgeRuntime` resolves `RobotBase.command_bounds()` into `SafetyGuard` before
constructing its executor. Resolve the existing command-safety profile tolerance
first, then intersect strict device bounds and the optional scalar position cap.
An empty intersection is a configuration error. `execution_limits()` supplies the
same effective intervals to validation and Smooth reset/step; it includes the
scalar cap and rate-check period for diagnostics. Smooth applies these resolved
bounds and the stricter of its own/shared rates after shaping, before updating
history. `bounded_step` reuses
`braking_velocity` and intersects feasible velocity intervals, including per-tick
acceleration. An infeasible state raises; final roundoff clipping does not repair
an infeasible increment. Stored velocity is the emitted position delta / dt.
Bounded holds retain this history; gripper command diagnostics reflect the final
bounded command. Without shared position bounds the original Smooth path is
unchanged. Direct/MPC retain their generation and final guard.

The shared runtime `_send_command` path covers warmup and formal ticks. On guard
or dispatch failure it writes `command_rejected` before re-raising, retaining
attempted commands, measured state, clocks, units/channels, plan/request IDs,
shared limits, dispatch progress and `_exception_detail` causes. A safety rejection
must not inherit the previous submission's dispatch progress. Failed attempts are
events only; `record_tick` still follows successful submission. Optional status or
coordinate lookup failures add `diagnostic_error` without losing the core attempt.
Recorder failure must not replace the original exception. Metadata saves
`command_limits`, with other executor-specific settings recorded separately.
User operation is described in [command limits](../../../../guideline.html#command-limits).

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

At rollout exit, stop the robot and policy worker, perform the configured Home
only after a successful stop on a non-fault exit, and close control resources
before video draining or Zarr finalization. Cleanup failures must leave
`run_completed: false` and their details in `result.json`; no successful finish
event is published when control cleanup fails. A late GUI cleanup error corrects
the saved result without rewriting trajectories or encoding videos again.

Idle recovery between rollouts (**Clear error**, **Return Home**, **Drag**, **Release**) is
scheduled by the `manimux run` session and implemented by the robot assembly. The
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

Camera diagnostics and record management use progressive disclosure: Images is
collapsed by default while scene previews remain visible. Experiment records shows
counts and readable session/rollout results; filters, bulk mutations and storage
details are collapsed. Each rollout row separates its name from result metadata;
expand the row for its full path and file listing. Camera and recording content
share an inset from their folder borders rather than using edge-to-edge text.
These display changes do not change recording ownership or deletion confirmation.

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

RPent with policy calls enabled uses this same optional activation flow. Its preview retains the selected
VLA prompt, horizon and runtime settings and reuses policy-call gripper and
calibrated path validation. The original operator pause state drives warmup;
RPent's idle hold must not hide Start. Dispatch remains disabled throughout
warming, draining and reset, so the managed native planner starts only after
reset acknowledgement and the next fresh observation. Reset failure, cancellation
or stale controls cannot enable it. Preview work creates no agent tool calls or
formal dispatched-step/chunk counts, and all timing and motion limits still apply.

An explicit `agent.debug_max_plan_age_s` overrides only managed GUI RPent's copied
plan-age limit for both preview and execution. It must be finite and positive. Provider
request timeouts, control freshness and motion checks remain unchanged. The opt-in
participates in the collection fingerprint; select a new debug variant for recorded
conditions. Omission preserves existing fingerprints and runtime behavior.

`agent.policy_enabled: false` selects observation and motion primitives only for
managed RPent. This boolean defaults to true. It omits the policy worker and VLA
warmup in that runtime. Runtime
status exposes the capability; `/execute` rejects disabled policy calls and the
native Toolkit omits both `execute_policy` and `observe.policy_contract`. Shared
motion, safety, control freshness and Start/Pause/Finish behavior stay unchanged.
Use a distinct recorded variant; `false` is fingerprinted, while omission and
`true` retain the baseline fingerprint.

Warmup never submits another request in the tick that decodes a response: inline
IK can block, so the next request must use state/cameras acquired on the next tick.
Otherwise the previous decode becomes observation age in the next E2E sample and
inflates delay calibration. Completed events include `observation_age_ms`,
`request_to_decode_ms`, `decode_ms`, and `post_decode_ms`; these sum to `e2e_ms`.
`inference_ms` is the worker model-call duration (including its client/transport),
a subset of request-to-decode time, not a separate additive term or GPU-only time.
RoboGUI labels E2E explicitly and displays model-call and decode/IK durations.

`PolicyWarmup` uses `PreviewPlanner` and the same `PlanningContext` as rollout.
The resolved config supplies the scheduler, strategy requests, decode seed/time,
source cap, trimming/blending, candidate selection, adaptive prefix, IK and executor.
Each preview owns its timeline, strategy/adapter, executor and command-safety history;
no robot or command sink is passed to it. The process decoder, when configured, is
borrowed exclusively and drained before backend reset. The physical runtime continues
to send its existing measured-position hold command, never preview commands.

Serial waits for its configured prefix to finish. Async progress/refill triggers and
RTC/PAINT condition generation use actual preview progress. AutoHorizon and candidate
samplers retain their specialized requests. There is no alternating synthetic branch
loop or ordinary-sampler fallback. Preview state uses measured feedback and real camera
frames; it does not simulate the robot achieving predicted positions. Consequently
warmup is a compute-path check, not a physical feasibility or task-success claim.

RTC/PAINT require one accepted ordinary plan and two accepted conditioned plans before
an early Start can complete draining. This follows their natural branch sequence, rather
than resetting history to manufacture another ordinary call. First calls remain visible
but are excluded from latency calibration; accepted responses update the preview
strategy's own estimator exactly as in rollout. Formal strategy history is reset and
only steady latency samples are transferred after backend reset acknowledgement.
Cold compilation may exceed the executable plan deadline: the warmed response is
reported as rejected and never bypasses configured age/deadline checks.

E2E ends at preview plan acceptance, matching formal plan timing. Step IK and executor
work occur on control ticks and are reported separately as `execution_ms`/`ik_status`;
they must not be mislabeled as GPU inference or whole-chunk decode. GUI preview cursors,
trimming and condition spans come from the runtime preview, not a free-running client
animation. Preview commands/plans are excluded from formal recordings and step counts.

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
substeps are rejected explicitly. Use the robot's native IK (`ik_backend: analytic`
where that adapter exposes the option); do not silently substitute it for diffIK.
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

### Model identity changes require a HELLO check

Adding a field to `backend_identity.fields` changes the runtime handshake contract.
A successful config loader or `--check` is insufficient: verify that the loaded
model's `runtime_metadata()` or the server's explicit metadata allowlist actually
publishes that field in HELLO_ACK. Prefer the loaded model's effective setting over
copying an expected value into the response. Exercise matching, mismatching and
missing fields through the HELLO builder and runtime comparator before providing
launch commands. Do not remove identity checks to make a missing-field error pass.

## Recording management

RoboGUI's `RecordingsPanel` browses the current method output directory using
`recording.management`. Operators select complete rollouts and confirm paths and a
reason. Deletion is reversible: whole episodes move to `_archive/deleted`, outside
inventory and repeat scans. `.deleted` files reserve the original rollout IDs;
the runtime allocator must not reuse them. Restore refuses to overwrite any newer
finalized or partial recording. Saved manifests and other layouts remain unchanged.

Serialize recording changes with Prepare and evaluation. Block mutations during an
active episode, startup, recovery, pending selection, evaluation/save, or a missing
service heartbeat. Recheck the metadata/result/label revision at confirmation.
Refresh repeat history and existing derived indexes after deletion/restoration.
Imported history is read-only in this panel. Test path boundaries, partial records,
changed labels, rollback, ID allocation, progress and GUI lifecycle interlocks.
