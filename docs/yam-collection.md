# YAM collection through ManiMux

This is a local copy of the YAM-ABC-Reproduce collection interface, not a new
Viewer screen. Collect and Review retain the original layout, camera previews,
task selection, recording controls, teaching-handle buttons, and episode format.
The original YAM-ABC-Reproduce checkout is not modified or imported.

## Start

Run from the repository root using the YAM environment. The optional
`collection` extra includes the GUI and video dependencies; hardware additionally
needs the existing ManiMux YAM/i2rt and camera SDK environment.
For a hardware-free GUI preview in a separate project environment:

```bash
uv sync --dev --extra collection
uv run python -m manimux.collection --mock
```

With the existing YAM environment:

```bash
envs/yam/.venv/bin/python -m manimux.collection --mock
```

Open **http://127.0.0.1:8043**. Without `--mock`, the same command enables the
hardware path, but follower control starts only when **Start Teleop** is clicked.
Start Teleop includes the original approximately one-second alignment ramp per
arm; it is not a preview-only operation. Grippers without pinned travel limits
may move during i2rt calibration. Support the arms before Reset Session or exit:
driver shutdown closes their motor connections, unlike Pause, which holds pose.

```bash
envs/yam/.venv/bin/python -u -m manimux.collection \
  --config configs/collection/yam/station.yaml \
  --cameras configs/collection/yam/cameras.yaml \
  --host 0.0.0.0 --port 8043
```

The default roster configures five RGB streams at 640×480, 30 FPS: **Top,
left wrist, right wrist, Gemini305 and Gemini335**.
An explicit five-camera roster is also available as
`--cameras configs/collection/yam/cameras-with-gemini.yaml`.
Match the serials to the local devices.
`--cameras` overrides the whole roster, not just the external view. Without it,
`station.yaml` uses its `cameras_config` field, which points to the same file.
Collection opens the devices directly. When switching from policy deployment,
finish the rollout, exit its runtime, and stop any separate camera server holding
the selected devices before opening collection previews.

Previews connect independently: a missing camera is hidden from the video grid,
while healthy cameras keep streaming. The rail reports the missing camera and its
error, and the CAM badge counts healthy streams against the configured roster.
A stream with no new frame for two seconds is hidden until capture resumes.
While the robot session is idle, **Preview** retries unavailable cameras without
reopening healthy ones. The selected camera roster stays intact; starting
collection or a new recording still requires every configured camera. To collect
fewer views deliberately, select a camera config containing only those views.

### Direct 100 Hz teleoperation with 30 FPS video

The current station config sets `collection_hz: 100`. Every nominal 10 ms,
the synchronous loop reads the latest leader positions and follower feedback,
submits the two leaders' targets together through the Direct executor, and saves
one command/state row per arm. It does not first sample at 30 Hz or interpolate
between old leader targets. Initial Start Teleop alignment remains a separate
operation. The default toggle gripper now sends its final open/close target
in one update; the common profile disables the former closing ramp.

In the GUI's Teleop controls, enter a positive **Hz** value (for example 30, 60,
80 or 100), then click **Apply Hz** or press Enter. Decimal values are accepted.
It applies between complete control steps while teleop continues, retaining the
current command, filter/limit history and gripper closing progress. Robot and camera
connections remain open. Stop recording and wait for saving to finish before changing
frequency; each episode keeps one configured rate. The next episode records the new
rate and timestamps. The selection survives browser refresh, Preview and Reset Session
within this GUI process, but does not overwrite the station YAML. Restarting the process
loads its YAML default. Online changes support synchronous Direct/Smooth collection;
the separate threaded configuration is not changed through this control.

`collection_hz` in `configs/collection/yam/station.yaml` sets the startup default.
Both target updates and command/joint recording follow the selected frequency;
the local leader policy interval, executor period and safety rate-check interval
are updated together. Cameras keep their own `fps` in `cameras.yaml`. Setting `collection_hz`
to `null` restores the existing 30 Hz profile timing and schema v1 recording;
setting it to `30` uses 30 Hz control with the new independent-camera format.
The override requires synchronous collection with `local_yam_leader`; it does
not edit the shared profile or change any learned policy's deployment interval.

After installing this UI/code update, restart the existing GUI once to load it.
Subsequent frequency changes use **Apply Hz** without restarting the process.

The GUI shows configured and actual loop Hz. These are application target updates,
not guaranteed per-motor CAN transmission or firmware control rates. The SDK still
sends its latest target through its own communication loop. Repeated joint values
are possible when motion is small or a newer feedback sample is not yet available.

With a numeric `collection_hz`, schema v2 saves independent streams:

| Saved stream | Rate / timestamp |
| --- | --- |
| `action-<arm>-joint/gripper.npy` | One latest leader target per control tick |
| `<arm>-joint_pos/gripper_pos.npy` | Follower feedback snapshot read before that tick's target |
| `controller-<arm>-joint.npy` | Executor target before driver joint-limit clipping |
| `controller-<arm>-timestamp-ns.npy` | Host command submission time, not CAN transmit time |
| `<arm>-feedback-timestamp-ns.npy` | Host driver snapshot time, not per-motor CAN receive time |
| `tick-timestamp-ns.npy`, `tick-monotonic-ns.npy` | Common control tick clocks |
| `<role>-images-rgb.mp4`, `<role>-timestamp.npy` | Each new camera capture, nominally 30 FPS |
| `<role>-frame-index.npy` | Latest capture at/before each control tick; −1 before the first image |

Both arms share a control-row count, but have separate command/feedback timestamps.
Camera counts are independent of that count and of each other. For example, ten
seconds ideally gives about 1,000 command/state rows per arm and 300 frames per
camera. Image copies run in the camera threads; the control loop does not wait for
a new image or copy camera images every 10 ms. Images are buffered in memory and
encoded when recording stops, as before.

`metadata.json → extra.timing` reports actual rates, counts and control/command
period statistics. `num_frames` means control rows in schema v2. Common host time
allows alignment but does not imply simultaneous camera exposures and motor reads.
GUI Review and Viser replay use the separate saved camera timestamps/frame maps.
ABC export preserves independent message times. LeRobot export explicitly builds
a uniform control-rate grid over the common interval: it holds the latest command,
state and image, so exported video contains repeated images. It rejects gaps longer
than three nominal periods and leaves the original 30 FPS videos unchanged.

### Per-command teleop timing diagnostics

The **Lead target · 分段耗时** panel shows the latest 200 cycles, including wall
time and the calling thread's CPU time. Three separate state-read scopes identify
the existing synchronous reads: `observation_left_arm`, `observation_right_arm`
and `precommand`. Each contains the left/right SDK wrapper calls; the reads,
their order, command values, force feedback and safety checks are unchanged.
`submit.left_sdk_submit` and `submit.right_sdk_submit` locate each arm's submission
inside the full `backend.follower_sdk_submit` span, which includes the target
sequence. Exceptions are recorded and propagated normally.

Complete diagnostics are buffered independently of **Record**. **Pause** saves
them in a background thread under
`<save_root>/.diagnostics/teleop/<timestamp>_<id>/control-timing.jsonl`, with
`metadata.json` and `write_complete.flag`. The panel and terminal show the saved
directory or save error. Loop failure and normal loop shutdown also flush the
buffer. Alignment and pause/hold are separate `kind` values, so their submissions
can be excluded from steady teleop statistics. Repeated calls within alignment
retain every timestamp in `spans`. Complete logs are not truncated to 200 cycles.
Diagnostic shutdown saving does not change any robot recovery behavior.

Schema 2 uses monotonic nanoseconds for each cycle and span offset/duration.
`previous_pacing` identifies the preceding cycle explicitly and records requested
sleep, actual sleep, excess sleep and the GUI's smoothed frequency. This avoids
changing an episode row after it has been frozen for saving. The final cycle's
sleep may have no following row; missing pacing is not zero sleep. Each episode
also retains its own `control-timing.jsonl` for its recorded samples.

Wall time minus thread CPU time includes lock/I/O waiting **and** time when the
thread was not scheduled; it does not identify an individual SDK lock. Nested
stages overlap and must not be added together. State wrapper calls are cache reads
that may wait on SDK locks; their timestamps are not motor feedback reception
times. SDK submission timestamps are not CAN transmit/receive acknowledgements.
Instrumentation itself has overhead; mock tests verify logging and command
preservation, not the real station's achievable frequency. Data remains in memory
until a flush, so forced process termination can lose unsaved diagnostics.

### Native joint-rate recording and reconstruction comparison

Enable **Native joint rate** beside the Record button before an episode. It is
also available as `record_native_joints: true` in the station YAML (default false).
The checkbox applies to both the GUI Record button and the teaching-handle record
button, and is locked while recording/saving. This adds diagnostic sidecars without
changing camera FPS, the configured teleop target rate, or the existing action fields.
It currently requires motorized YAM teaching arms and YAM followers; unsupported
drivers, passive GELLO and mock robots report an error instead of fake native data.

Each completed episode additionally contains:

```text
native_joints/
  metadata.json
  leader_left.jsonl
  leader_right.jsonl
  follower_left.jsonl
  follower_right.jsonl
```

Independent read-only SocketCAN listeners receive each new motor feedback packet.
They never send commands or call the SDK's cached-state getters. Each JSONL row
contains `joint_index` (0–5), `sequence`, `timestamp_ns`, `position_rad`,
`velocity_rad_s`, `effort_nm`, and `motor_error`. The six arm joints are measured
states, not commanded actions; the gripper is excluded. Joint calibration and
motor decoding come from the active i2rt driver. Timestamps are **host kernel CAN
receive times in Unix nanoseconds**, not motor-internal sampling times. Different
joints in one bus sweep have different timestamps. The metadata records calibration,
sample counts, socket drops and capture errors. `complete: false` means the native
sidecar cannot be treated as a complete capture, even if the ordinary episode saved.
Writers run independently of the controller, and stop before episode finalization.

After recording, run the offline comparison (no robot connection):

```bash
uv pip install --python envs/yam/.venv/bin/python 'matplotlib>=3.8'
envs/yam/.venv/bin/python -m manimux.collection.yam.data.joint_rate_analysis \
  /absolute/path/to/episode
```

Output goes to `episode/joint_rate_analysis/`: one PNG and SVG per arm, with a
position overlay and error curve for every joint; per-joint curve CSVs;
`metrics.csv` with RMSE, MAE, 95th percentile absolute error and maximum absolute
error (degrees); and `summary.json` with sampling parameters and quality flags.

The reference is native feedback sampled at 100 Hz, using the latest feedback at
or before each tick. The candidate first samples the **same trajectory** at 30 Hz,
then linearly interpolates to 100 Hz. A third curve adds the legacy executor's
8 Hz first-order low-pass, without motion limits; `--cutoff-hz 0` omits that curve.
There is deliberately no anti-alias prefilter before the 30 Hz sampling, matching
the current teleop sampling behavior. `--target-hz`, `--low-hz`, and `--phase-ms`
control the grids; phase defaults to the episode start. Non-increasing timestamps
are rejected, invalid motor feedback is masked, and gaps over `--max-gap-s 0.05`
are excluded. Interpolation never extends beyond available lower-rate endpoints.

This measures lost trajectory detail and reconstruction error, not the physical
tracking error of running a different controller. Recording at native rate does
not make the follower execute 100 Hz leader updates; that is a separate control
experiment. Existing 30 Hz-only episodes cannot supply the missing reference.

<a id="synchronized-viser-replay-video-linear-100-hz-held-30-hz"></a>

## Start Teleop appears unresponsive

Check the terminal running the collection GUI, or its redirected stdout/stderr
log. A click can reach `POST /api/collect/start-teleop` and fail with HTTP 500
before teleoperation starts. One observed startup failure was:

```text
AssertionError: fail to communicate with the motor 1 on yam_real at can channel socketcan channel 'can_left'
```

This means initialization received no response from motor 1 on the configured
left-follower CAN channel. It does not identify the physical cause by itself.
Check the selected follower/channel mapping, arm power and CAN connection.
An interface shown as `UP` by `ip -brief link show type can` only confirms the
interface state; it does not prove that the motor responds.

The current handler catches `RuntimeError` but not this i2rt `AssertionError`.
The frontend expects JSON, so the plain-text HTTP 500 response can fail parsing
without showing the error toast. This is an error-reporting limitation, not a
successful connection. Inspect status without issuing another motion request:

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8043/api/collect/status \
  | envs/yam/.venv/bin/python -m json.tool
```

Successful teleoperation should report `live: true`, `teleop_running: true`,
both arm feedback streams in `arms`, and `last_error: null`. A failed startup can
leave `live: false`, `teleop_running: false`, empty `units`/`arms`, yet still show
`last_error: null` because the teleop loop was never built. Use the server traceback
to diagnose that case. Five working camera previews establish camera readiness
only; they do not establish robot connectivity or a saved recording.

## Control and configuration

```text
Original GUI / teaching-handle buttons
  → YamLeaderPolicy (collection_hz, radians + continuous normalized gripper)
  → atomic dual-arm target update
  → synchronous executor (same Hz) OR separate legacy threaded configuration
  → ManiMux SafetyGuard → YamDualArmDriver → i2rt → followers
```

- `configs/collection/yam/station.yaml`: leader/follower mapping, leader calibration,
  bilateral feedback, analog/toggle trigger, toggle closing duration, task and output directory.
  Follower hardware settings come from the control profile. `collection_hz` overrides
  collection timing; without it the action rate comes from that same profile.
- `configs/collection/yam/cameras.yaml`: default Top, two wrists, Gemini305 and Gemini335 roster.
  GUI camera edits write the selected roster file.
- `configs/collection/yam/cameras-with-gemini.yaml`: explicit five-camera roster;
  select with `--cameras`.
- `configs/collection/yam/control.yaml`: standard ManiMux execution/robot config.
  Default `execution_mode: synchronous` runs one executor step and sends one
  dual-arm command per target update, with no extra command thread. The station
  override currently selects 100 Hz; `collection_hz: null` uses the profile's 30 Hz.
  `executor: direct` adds no filtering; the default common arm rate limits are disabled.
  The shared gripper closing limit is disabled: toggle sends 0 (close) or 1 (open)
  directly, without the former one-second closing curve.
  Finite common motion limits also apply to Direct. Hardware joint limits,
  PD gains, motor capabilities and configured command-safety limits still apply.
- `configs/collection/yam/station-threaded.yaml` selects `execution_mode: threaded`
  and `control-threaded.yaml`: 30 Hz target updates, 100 Hz execution. These rates
  are configurable, not hard-coded. Threaded execution can also run at 30 Hz.
  Synchronous execution requires station and robot control frequencies to match.
- To collect with inference-time smoothing, select `execution.executor: smooth`
  and the same filter settings. Motion limits come from the shared profile for both
  executors. Existing inference configs without a profile are not changed automatically.
- The installed ManiMux i2rt fixes gripper force at **50 N**; this copy uses 50 N,
  not the original station's 60 N. Unsupported force settings fail explicitly.
- Leader inputs time out after `max(policy.timeout_s, 2 × target period)`.
  The configured floor defaults to 0.5 s: 30/100 Hz retain a 0.5 s timeout,
  while 1 Hz uses 2 s so normal one-second updates are not rejected. During a
  frequency switch, the preceding target's interval also applies until the first
  fresh target arrives. An already expired target cannot be revived by changing Hz.
  `extra.manimux.leader_timeout_s` records the effective timeout. Failure latches
  a stop in threaded mode. Synchronous mode has no independent watchdog thread:
  an expired target is rejected when publication resumes, but a blocked sampling
  thread cannot itself issue a stop. Neither mode detects a frozen encoder that
  continues returning plausible readings. It is not a certified safety stop.
- Pause disables publication and holds measured pose. E-stop latches follower
  commands off; reset/rebuild is required. No automatic CAN reset or Home occurs.

The copied Train/Deploy, global CAN reset, motor power-off and process-kill
endpoints are disabled. Use ManiMux's existing training tools and `manimux serve`
for those workflows. The GUI defaults to localhost and has no authentication;
do not expose it directly to an untrusted network.

## Code boundaries

`src/manimux/collection/cli.py` reads `collector: yam` and dispatches to
`src/manimux/collection/yam/cli.py`. The YAM folder owns its GUI, station config,
leader, control loop, execution backend, cameras and recorder. A future embodiment
can register its own config-driven entry point without adopting the YAM GUI,
dual-arm layout, gripper convention, or on-disk format.

`manimux_config` references a regular ManiMux config; it can point to an inference
config to reuse its robot/control frequency, executor and command-safety settings.
No policy worker, network model server or inference scheduler is started by the
collector. With a control profile, station or GUI hardware overrides must match it;
legacy configs without a profile retain station-supplied overrides. Collection
always disables automatic Home. Match and inspect the resolved episode metadata,
not just the filename, when claiming collection/inference control equivalence.

## Shared control profile

`configs/robots/yam/common.yaml` is the single shared source for the YAM driver,
group dimensions, follower CAN channels, arm/gripper types, gripper force and
action-point interval. The YAM driver retains its six-joints-then-gripper order,
radian joint units, and normalized gripper convention (0 closed, 1 open).
This does not introduce a second generic robot or calibration framework.

The standard `manimux.config.load_config()` resolves one `control_profile` reference,
relative to the declaring config file (absolute paths also work). Profiles cannot
inherit other profiles. Existing paths such as `robot.config` and
`robot.options.right_config` keep their original working-directory-relative semantics.
Local values may repeat shared values, but conflicting values are rejected rather
than silently overridden. `trajectory_duration_s` cannot change the shared effective
action interval. The collector fills omitted hardware/rate fields from this same loader
and rechecks GUI-supplied hardware values before connecting.

Current consumers are `control.yaml`, `control-threaded.yaml`, and both
`configs/pi05/yam/infra/put-bottles/rtc-joint-step30000.yaml` and
`configs/pi05/yam/infra/put-bottles/rtc-joint-ee-step30000.yaml`. Other inference
configs remain unchanged; they do not automatically inherit this profile.

`command_safety: null` explicitly disables the *additional shared software envelope*,
not hardware protections or Smooth's shaping limits. To enable a common envelope,
use the existing `CommandSafetyConfig` format: `position_lower`, `position_upper`,
`max_velocity`, and `max_acceleration`, each mapping group names to per-joint vectors.
All four are required together. Values use radians, radians/s and radians/s² for
arm joints, and normalized aperture units for the gripper. Both Direct and Smooth
are checked after execution; violations stop/reject commands, not silently clip them.

Normal command shaping limits are now also owned by `common.yaml`:

```yaml
motion_limits:
  arm:
    max_velocity: null
    max_acceleration: null
  gripper:
    group_indices: {left_arm: 6, right_arm: 6}
    max_velocity: null
    max_acceleration: null
    max_closing_velocity: null
```

Each `null` independently disables that software rate constraint. A finite positive
value constrains command velocity or acceleration in both Direct and Smooth;
gripper limits are independent of arm limits. The default `max_closing_velocity: null`
disables gradual closing in both the leader toggle policy and Direct executor:
one press changes the target directly to 0 or 1. For an explicit slow-closing
experiment, a finite value such as 1.0 limits decreasing normalized aperture
to one unit per nominal second; missed control deadlines stretch that ramp's
wall-clock duration. Actual physical closure still depends on tracking, contact and force.
The config loader resolves these into
Smooth's effective fields and rejects conflicting local limits. Direct uses the
same limiting functions without the low-pass filter. Existing configs without a
shared motion section retain their original executor defaults. This shared section
currently supports Direct and Smooth only; MPC is rejected rather than silently
ignoring the gripper/arm split.

Executor selection, output frequency, Smooth filtering, inference chunk scheduling
and collection toggle-button behavior stay in their own configs. Defaults remain
synchronous Direct collection and 100 Hz Smooth RTC, with 50 N gripper force.
The shared profile's action interval is 33.33 ms; the station's `collection_hz: 100`
override uses 10 ms for collection only. The migrated RTC config no longer
imposes its former arm limits of 0.25 rad/s and 0.5 rad/s². Its former symmetric
gripper velocity/acceleration limits and the shared closing-only limit are disabled.
Its 8 Hz arm low-pass filter, position bound and gripper aperture bounds remain.
Collection's toggle-close duration is derived from the shared closing speed;
with no closing limit it resolves to zero and Direct sends the final target.
Every config referencing this profile inherits that rate-limit change on reload;
it does not change model trajectory interpolation or the SDK's 50 N force control.
With no acceleration cap, a Smooth hold freezes the previous command immediately
instead of applying a bounded deceleration ramp. Hardware behavior is not inferred
from that software command; validate changed limits on hardware before judging quality.

The resolved profile path and effective hardware/execution values are saved with
episodes. Editing a profile does not hot-update a connected robot; finish the
session and restart the relevant service to apply it. No running service is
automatically restarted by a config edit.

Switch modes by selecting the station config:

```bash
# Default: synchronous 30 Hz, DirectExecutor
python -m manimux.collection --config configs/collection/yam/station.yaml --mock

# Optional: 30 Hz targets, independently executed at 100 Hz
python -m manimux.collection --config configs/collection/yam/station-threaded.yaml --mock
```

## Recorded evidence

Episodes default to `data/collection/episodes/<task>/<episode>/` and preserve
the original state/action `.npy` files, RGB MP4 streams, optional FK fields,
`metadata.json` and `write_complete.flag`. Original `action-*` fields remain the
leader-policy targets, not smoothed executor outputs.

`metadata.json → extra.manimux` records the resolved robot/executor configuration
and whether mock hardware was used. `manimux-control.jsonl` records each active
executor tick (one per target in synchronous mode): target, executor command, observed feedback, source sequence and
timestamps. `controller-*` arrays are latest executor outputs **before driver
joint-limit clipping**, not measurements of the native MIT controller input.
Feedback Unix timestamps are estimated from the ManiMux monotonic state clock.
Use these distinctions when choosing training targets or comparing tracking.

E-stop retains the original GUI behavior of discarding an unfinished episode.
Stopping recording writes the completeness marker only after arrays, video,
metadata and trace are saved. No upload or training job is started automatically.
