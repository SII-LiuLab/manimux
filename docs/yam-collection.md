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
envs/yam/.venv/bin/python -m manimux.collection \
  --config configs/collection/yam/station.yaml
```

After reinstalling the editable package, `manimux-collect` is the equivalent entry
point. Mock regression tests and a local live YAM collection have been exercised;
this does not establish timing or hardware equivalence for every configuration.
Do not run the original collection GUI concurrently. The new GUI shares ManiMux's
`yam` ownership lock with inference, but the original GUI does not honor that lock.

## Control and configuration

```text
Original GUI / teaching-handle buttons
  → YamLeaderPolicy (30 Hz, radians + continuous normalized gripper)
  → atomic dual-arm target update
  → synchronous executor (30 Hz) OR threaded executor (configured rate)
  → ManiMux SafetyGuard → YamDualArmDriver → i2rt → followers
```

- `configs/collection/yam/station.yaml`: leader/follower mapping, leader calibration,
  bilateral feedback, analog/toggle trigger, toggle closing duration, task and output directory.
  Follower hardware settings and the sample rate come from the control profile.
- `configs/collection/yam/cameras.yaml`: independent copy of the original camera
  roster, including optional Orbbec views. GUI camera edits write this copy only.
- `configs/collection/yam/control.yaml`: standard ManiMux execution/robot config.
  Default `execution_mode: synchronous` runs one executor step and sends one
  dual-arm command per 30 Hz target update, with no extra command thread.
  `executor: direct` adds no filtering; the default common arm rate limits are disabled.
  The shared gripper closing limit preserves the original toggle-close curve.
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
- Leader inputs time out after `policy.timeout_s` (default 0.5 s); failure latches
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
    max_closing_velocity: 1.0
```

Each `null` independently disables that software rate constraint. A finite positive
value constrains command velocity or acceleration in both Direct and Smooth;
gripper limits are independent of arm limits. `max_closing_velocity: 1.0` limits
only decreasing normalized aperture: a full 1→0 target transition takes one second
at the nominal control rate, while 0→1 is an immediate target switch. This matches
the original YAM toggle collection configuration; it is not a symmetric gripper
speed cap. Actual physical closure still depends on tracking, contact and force.
The config loader resolves these into
Smooth's effective fields and rejects conflicting local limits. Direct uses the
same limiting functions without the low-pass filter. Existing configs without a
shared motion section retain their original executor defaults. This shared section
currently supports Direct and Smooth only; MPC is rejected rather than silently
ignoring the gripper/arm split.

Executor selection, output frequency, Smooth filtering, inference chunk scheduling
and collection toggle-button behavior stay in their own configs. Defaults remain
synchronous 30 Hz Direct collection and 100 Hz Smooth RTC, with the same 33.33 ms
action-point interval and 50 N gripper force. The migrated RTC config no longer
imposes its former arm limits of 0.25 rad/s and 0.5 rad/s². Its former symmetric
gripper velocity/acceleration limits are replaced by the original collection's
closing-only limit, not removed altogether. Its 8 Hz low-pass filter, position
bound and gripper aperture bounds remain. Collection's toggle-close duration is
derived from the shared closing speed; the policy generates the original ramp and
Direct accepts that ramp without slowing it again. Both infer and collect use the
same closing contract. Legacy analog collection requires its own explicit profile
if it should follow the trigger without this closing constraint.
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
