# drivers/ — hardware boundary layer

Design rationale, empirical data, and known pitfalls for the modules that
talk directly to the arm SDK, the gripper CAN bus, and the XR headset.
Code comments in this layer are intentionally terse; this file is where the
"why" and the incident history live.

## arm_driver.py

Owns "how to safely push joint angles into the controller." What angle to
send is `retarget`/`ik_solver`'s job; whether to send it at all is
`safety`'s job.

### Three hard constraints (from SDK docs + incidents hit in `tools/`)

1. **Only one `Marvin_Robot` connection per process.** The SDK holds the UDP
   port exclusively. Clicking Disconnect in the MarvinPlatform GUI does not
   release it — the app must fully exit. The gripper reuses the same
   connection (CAN pass-through over the end-effector channel), so
   `gripper.py` must reuse `RobotConnection.robot` rather than connecting
   again — a second connection fails outright.
2. **High-rate commands must be bracketed by `clear_set()`/`send_cmd()`.**
   Per SDK docs, the command buffer refreshes at 1kHz and parameter writes
   only take effect between these two calls. Both arms must be dispatched
   inside the *same* `clear_set()`/`send_cmd()` pair to guarantee same-frame
   consistency.
3. **Every exit path must disable the servos.** Crash, Ctrl+C, or any
   exception — carried over from `tools/gripper_cycle.py`'s
   "unconditional disable in `finally`" habit.

`dry_run=True` runs the full logic path but skips `send_cmd()` — this is the
primary mode during the P2 validation phase.

### `RobotConnection.__init__`

- Right after a controller restart, `VERSION` reads back `0` and the
  end-effector CAN channel + realtime frame feed are not yet up. This must
  not be mistaken for a hardware fault — hit in `tools/wait_ready.py`.
- A "successful" connect can still be firewall-blocked at the UDP data
  level. The constructor samples `frame_serial` 5x over 50ms and requires
  at least 3 distinct values to consider the data channel alive.

### `ensure_clear` / `ERR_CODE_CN`

Shared by `scripts/home_now.py` and `scripts/goto_joints.py` (both call it
before `prepare()`) — moved here from being `home_now.py`-local once
`goto_joints.py` needed the same thing, rather than duplicating it: it's
driver-state logic (`STATE_ERROR`, `driver.state()`,
`conn.robot.clear_error()`), not a generic utility, so it lives next to
`send_joint_commands`/`move_to_joints` — the codebase's existing pattern
for cross-script driver operations (see `goto_joints.py`'s comment on
`move_to_joints`: "run_teleop's auto-home uses the same function, don't
duplicate it here").

`RobotConnection.check_and_clear_errors()` only fires `clear_error()` once
with no confirmation — not enough right after an e-stop, where err_code=13
(急停/Emcy) can still be latched even after the physical button is
released (SDK README: "急停后是自动下伺服的，需要清错再重新上伺服状态").
`ensure_clear()` retries (default 5x, 0.3s apart) and re-reads state each
time, only returning success once `err==0` and `cur!=STATE_ERROR` are
actually confirmed. Callers that skip this and go straight to `prepare()`
get a `RuntimeError` from `prepare()`'s own fault check instead of a clear
diagnostic — this is what fixed `goto_joints.py` raising a bare
`prepare()` traceback on a latched e-stop instead of clearing it first.

### `ArmDriver.prepare()` / `_check_vel_ratio()`

Every upstream rate limit in this codebase (see config's speed-budget
section) is derived under the assumption "the controller actually runs at
`VEL_RATIO`%". `_check_vel_ratio()` reads back `joint_vel_ratio` after
`set_vel_acc()` and after `set_state()` to confirm that assumption holds —
this closes the loop on a question that used to be open: does the per-frame
`clear_set()` used for joint commands wipe out the vel/acc set earlier?
(Answer: no, confirmed by the readback.)

If the assumption is silently false, upstream clamps end up *wider* than
what the controller enforces, tracking error accumulates, and you get
`tracking_error` latch faults — this was the P0-2 bug class.

Readback semantics: if `vel_ratio`/`acc_ratio` read back out of range
(0 or out-of-bounds), that's a warn-only — the SDK doesn't document whether
`RT_IN` always populates these fields, so an unreadable value shouldn't
block the operator. A readable-but-mismatched value is a real fault and
must block.

### `ArmDriver.prepare()` — torque/impedance branch

When `cfg.ARM_STATE==STATE_TORQUE` (set by `run_teleop.py --control-mode
impedance`), this branch requires an `impedance_config` (an
`algos.solver_config.ImpedanceConfig`, see docs/algos.md#impedanceconfig)
passed in at `ArmDriver.__init__` — raises immediately if `None`, same
"fail fast at construction, not mid-run" style as `ArmChannel`'s
`diff_ik_config` check for `solver='diff'`. `type=3` (force control) is
rejected outright: it needs a different call sequence
(`set_force_control_params`/`set_force_cmd`) that isn't implemented here.

`type=1` (joint) and `type=2` (Cartesian) share the same call sequence:
`set_impedance_type()` then **both** `set_joint_kd_params()` and
`set_cart_kd_params()` unconditionally (mirrors the MarvinPlatform GUI,
which saves both panels together; the inactive row — joint's or
Cartesian's, whichever `type` isn't — is harmless to send), all inside
the same `clear_set()`/.../`send_cmd()` batch as `set_vel_acc()`, then
falls through to the normal `clear_set()`/`set_state()`/`send_cmd()`
mode-switch that follows.

`type==2` gets exactly one additional call in that same batch:
`set_EefCart_control_params()` (`OnSetEefRot_A`/`_B`) — found while
debugging why Cartesian misbehaved on hardware after joint validated
cleanly (see docs/algos.md#impedanceconfig's `rot_type` section for the
full story). `set_cart_kd_params` (what both `type`s call, for whichever
row) has no end-effector-rotation-reference parameter at all; the
vendor's own demo purpose-built for this exact scenario
(`DEMO_C++/showcase_eef_cart_impedance.cpp`) always pairs
`OnSetCartKD_A` with `OnSetEefRot_A` in the same batch, and the original
code here only ever called the former — that gap is what caused the
"other joints doing something unexplained under end-effector force"
symptom.

**A now-reverted earlier attempt at this fix routed `type==2` through
`Concise_Marvin_Robot.set_imp_cart_state()` instead** — a single call
that bundles vel/acc + K/D + rotation-reference + the mode switch itself,
matching a *different* vendor demo
(`DEMO_PYTHON/showcases_new_control_sdk.py`'s `case5_impedance_cart`).
Never ran: `Concise_Marvin_Robot` is a separate class from `Marvin_Robot`
(the one `RobotConnection`/`ArmDriver` actually hold) with its own
independent `.so` binding and its own `Connect()` — confirmed locally
that `Marvin_Robot` has no `set_imp_cart_state` attribute at all
(`AttributeError`) before this ever reached hardware. Even if it had been
called on the right object, instantiating a second `Concise_Marvin_Robot`
connection alongside the existing one would risk the SDK's documented
one-connection-per-process UDP-port exclusivity (see this file's "Three
hard constraints" section above). `set_EefCart_control_params` is on
`Marvin_Robot` itself — no second connection, no new class, just one more
call in the batch this codebase already sends.

Entirely skipped under `--dry-run` — `prepare()` returns before reaching
this branch (same early-return that skips `set_state`/`set_vel_acc`). So
`--dry-run --control-mode impedance` never exercises these SDK calls at
all; only real hardware does.

### `ArmDriver.disable()`

`set_state()` is asynchronous — `prepare()` sleeps 1.0s after a mode switch
for it to take effect. Callers' `finally` blocks call `disable()` then
immediately `conn.close()` → `release_robot()`. Without waiting and
confirming disable, the connection can close before the controller has
processed the command. This was observed in practice: "已下伺服" was
printed, but `wait_ready.py` still showed arm A in state 1 — i.e. the motor
was still energized. This is a safety issue, not a logging one. `disable()`
retries up to 3x with a 0.3s wait and state readback before giving up.

### `send_joint_commands` / `move_to_joints`

Multi-arm commands are dispatched inside a single `clear_set()`/`send_cmd()`
pair. For coordinated tasks, dispatching the two arms one control period
apart shows up directly as skew in the grasped object.

`move_to_joints()` (shared by `goto_joints.py`'s manual moves and
`run_teleop.py`'s auto-homing on exit — do not fork a second copy, they will
drift):

- Starts from the **measured** position, not a stale commanded one, to
  avoid a jump at enable.
- Cosine ease-in/ease-out (S-curve), default peak 8°/s, well below teleop
  speeds. All axes share one `frac(t)` of the same time base, so the move
  is a straight line in joint space — no axis reaches its target early
  and sits idle while others are still catching up, which the old
  constant-rate-per-axis version could do. Velocity is 0 at both
  endpoints instead of stepping instantly to full speed. `speed_deg_s` is
  the *peak* per-axis rate (at the move's midpoint), not the average, so
  total time is ~1.57x (`pi/2`) the old constant-rate estimate for the
  same distance — accepted deliberately since homing favors smooth over
  fast (see `docs/core.md`).
- Compares commanded vs. measured every 0.1s; aborts immediately on
  excess error (e.g. hit an obstacle) rather than pushing through.
- Ctrl+C stops in place and returns; the caller is still responsible for
  disabling the servos.
- **The 3rd return value is where the arm actually stopped**, not the
  requested `targets` — on interrupt/timeout these differ. A caller that
  wants to hold position after must use this value; using `targets` instead
  turns "stop in place" back into "snap to target," which defeats the
  safety semantics.
- Timeout fallback = 2x the nominal duration + 5s. Without this, a joint
  that never converges (limit contention, servo fault) turns this into an
  unbounded loop.

### `ControlLoop`

Fixed-rate control loop skeleton; business logic is injected via the `step`
callback. `before_disable` runs after the loop stops but **before** the
servos are disabled (homing hooks in here, since the motors must still be
energized to move). Any exception from `before_disable` — including a
second Ctrl+C during homing — is swallowed so the unconditional `disable()`
call after it is never skipped.

On a missed tick, the loop resets its timing baseline instead of trying to
catch up, which would otherwise burst multiple commands back to back.

## gripper.py

Drives an OmniGripper (DM4310, made by 达妙) over Marvin's end-effector CAN
channel 1.

**Protocol implementation is reused as-is from `tools/km_gripper.py`**
(already debugged, already had its incidents) — `f2u`/`u2f` scaling, the
`0x7FF` register read, and the MIT-mode packing all import from there rather
than being reimplemented, to avoid the two copies drifting apart.

The only real change here is **connection reuse**: `km_gripper.Gripper`
normally calls `Marvin_Robot().connect()` itself in `__init__`, but in the
teleop pipeline the arm driver already owns that connection, and the SDK's
UDP port is exclusive — a second `connect()` fails outright. So this module
takes an already-connected `Marvin_Robot` instance and only handles ctypes
prototype registration and send/receive.

Carries forward all four protections from `tools/gripper_cycle.py`,
unchanged:

1. Zero-error start — first command after enable is issued at the measured
   position, so there's no bounce at enable time.
2. Slew-rate limiting — position error stays bounded at all times.
3. Dual-threshold abort — torque or position-error over limit stops and
   disables immediately.
4. Unconditional disable in `finally` — crash or Ctrl+C still de-energizes.

Gripper I/O must not run inside the 250Hz joint control loop: CAN
pass-through and arm commands share one UDP socket, and `send()` retries up
to 30ms when the previous frame hasn't drained — folding that into the
joint loop would tank tracking. `GripperThread` runs it on its own 50Hz
thread instead.

### `GripperThread.start()`

Zero-error start ordering matters: the DM4310 in MIT mode only replies when
it receives a frame — `state()` passively reads the CAN buffer and issues no
request of its own. So reading **before** enable always returns `None`;
that's expected, not a fault. The sequence
(`enable → sleep(0.15) → state(timeout=0.3)`) is copied from the already
debugged `tools/gripper_cycle.py:85-89`. Passing `tools/probe_gripper_alive.py`
does **not** imply this path will read data — that probe uses
`read_rid()`, a 0x7FF active register request, a different path entirely.

The target is also initialized to the current position rather than left at
its `0.0` default — otherwise the thread would immediately slew toward
`GRIPPER_OPEN_RAD` the moment it starts, and those two constants are not
yet calibrated (direction could be into the mechanical limit). Starting at
zero motion means the gripper only moves once the operator pulls the
trigger.

### `GripperThread._run()` — force-limited position control

The commanded position is clamped to within `GRIPPER_MAX_OVERSHOOT_RAD` of
the last measured position. When the gripper is holding an object and the
measured position stalls, the command gets pinned at `q_meas ± overshoot`,
so grip force settles at `KP × overshoot` instead of climbing until the
torque protection trips. Self-correcting: clamped value is written back to
`self._cur`, so the command keeps advancing once the object is removed.
Same idea as the arm side's "start every frame from the actual pose."

## xr_source.py

ctypes binding straight onto `libPXREARobotSDK.so`, reading PICO controller
data via `roboticsservice`. No pybind11 layer needed — the `.so` exports
only 4 C functions with no missing dependencies, so `ctypes.CDLL` loads it
directly.

Startup order (reversed order fails to connect):
1. Host: `/opt/apps/roboticsservice/runService.sh`
2. PICO: open XenseVR-Toolkit, check Controller, enable Send

### Design constraints

- The SDK callback runs on its own thread. The callback body only does
  `json.loads` + writes into a slot — no IK, no networking, no printing,
  or it stalls the data link.
- The `CFUNCTYPE` instance must be kept as a strong module-level reference
  (`_CB_REF`); if it gets garbage collected, the process segfaults.
- `latest()` returns a snapshot; callers can hold and use it freely without
  it being mutated by the next frame.

### `parse_pose` — measured axis convention

Session B, 2026-08-12, measured on real PICO 4U hardware (not assumed from
Unity docs): **+X = right, +Y = up, +Z = toward the body (back) / −Z = away
from the body (front)**. This is a standard right-handed frame (X×Y=+Z
matches the measured orientation) — earlier code comments describing this
as "Unity's left-handed +Z-forward" were a pre-hardware assumption and were
wrong. Full derivation: `tianji_teleop_plan.md`, Session B. Converting into
the robot base frame is `retarget.py`'s job; this module always returns the
raw measured values unmodified.

### `_on_callback` — envelope unwrapping

The measured `stateJson` payload is a one-layer envelope:
`{"functionName":"Tracking","value":"<json string>"}`. Docs/planning
materials had assumed `stateJson` was directly the `Head`/`Controller`
structure; that assumption didn't match the real payload. The callback
unwraps this envelope and re-parses `value` (itself a JSON string) to reach
the actual tracking data.

### Recording path

Recording only appends `(host_ns, dev, raw)` tuples onto a `deque` from the
callback; a separate writer thread drains it to disk. `deque.append` is
O(1) and thread-safe, so the callback stays minimal. An earlier version had
the recording thread poll `latest()` every 2ms and dedupe — when frames
arrived faster than the poll, it silently dropped them: measured 3288
frames in, only 2664 recorded (19% loss). The queue-based design does not
lose frames.

## umi_source.py

Turns a recorded UMI episode (LeRobot v3, from xense-taccap-lerobot's
`bi_taccap_gripper`) into a frame stream `algos.retarget.Retargeter` accepts
unchanged. It exists so a recorded demo runs through the *same* lowpass →
clutch → Cartesian-limiter → IK → safety-gate chain as live teleop, rather
than through a parallel replay path that could drift from it.

### Why it can impersonate `XRFrame`

`Retargeter.update()` only ever calls two methods on a frame: `pose(hand)` and
`button(hand, name, default)`. `UmiFrame` implements exactly those two, so
nothing downstream needs a code path for "recorded" vs "live".

`button('grip')` returns 1.0 unconditionally — the clutch reads as permanently
pressed. A recorded episode has no clutch: the demonstrator's intent for every
frame in it was "follow me". Frame 0 therefore engages immediately, latching
onto the robot's *measured* pose.

### The recorded format

| field | meaning |
|---|---|
| `{side}_tcp.x/y/z` | position, **meters** |
| `{side}_tcp.r1..r3` | first **column** of R |
| `{side}_tcp.r4..r6` | second column (6D rotation, Zhou et al.) |
| `{side}_gripper.pos` | normalized jaw, **0 = closed, 1 = open** |

Column indices are resolved from `meta/info.json`'s `features.action.names`
rather than hardcoded, so a layout change fails loudly instead of silently
reading the wrong axis.

The 6D rotation is orthonormal to ~4e-8 as recorded; `rot6d_to_mat` still runs
Gram-Schmidt, as a guard rather than a fix. `mat_to_quat` is the inverse of
`retarget.quat_to_mat`, round-tripping to 7e-16 (`--self-test`).

### `action` vs `observation.state`

`action[t]` is **bit-identical** to `observation.state[t+1]` — verified across
the whole `replay_test` episode, mean absolute difference exactly 0 on all 20
dimensions. The rig is a passive recorder (`send_action()` is a no-op), so its
"action" is just the next measured pose. Loading either gives the same
trajectory one frame apart; `load_episode` defaults to `action` so frame i's
target is the pose the demonstrator actually reached.

Worth knowing for anything downstream of this repo too: a policy trained on
these actions is predicting next-pose-in-VR-world-frame, and that frame's
origin moves every VR restart.

### Coordinate frame — and why it needs no new calibration

Poses are in the **Pico VR world frame**: X forward, Y left, Z up,
gravity-aligned, origin = the headset position when the Unity app started.
Not the raw PICO frame — `Pico4TrackerReader` already applied
`PICO_TO_WORLD_R` (`[x,y,z] → [-z,-x,y]`).

So `config.AXIS_MAP_UMI` is not a second, independent calibration; it is the
existing `config.AXIS_MAP` composed with that known remap:

```
v_base  = AXIS_MAP · v_pico
v_world = G · v_pico            (G = PICO_TO_WORLD_R)
=>  v_base = (AXIS_MAP · Gᵀ) · v_world
```

Checked numerically: `AXIS_MAP_UMI` equals `AXIS_MAP @ G.T` to 0.0 for both
arms, both determinants +1. It inherits `AXIS_MAP`'s status exactly — arm A
traces to a measured mapping, arm B to a derived and still unverified one. A
wrong-looking arm B result should suspect `AXIS_MAP['B']` first.

That the origin drifts per session is precisely why this feeds the
relative-mapping retargeter instead of being commanded absolutely.

### `resample()` — interpolation, not zero-order hold

`replay.py`'s `resample_to_control_rate` is a ZOH, which is fine for 90Hz XR
recordings. These are 30Hz. At 30→250Hz a ZOH hands the Cartesian rate limiter
a ~15mm step every 33ms; at this demo's peak speed that step needs ~8 of the
8.3 control periods per recorded frame just to be caught up, so the limiter
would sit permanently saturated and every reported `cart_lag` would be an
artifact of the resampler rather than of the trajectory. Lerp on position,
slerp on orientation instead.

The jaw value is held, not interpolated: it is a normalized position target,
and smoothing it buys nothing the follower's own impedance loop doesn't
already do.

### `slice_frames()` — recorded indices, sliced before resampling

`--frames A:B` (on `replay.py --umi` and `scripts/umi_replay.py`) takes
**recorded** frame indices, the ones a LeRobot dataset is indexed by and the
ones `scripts/umi_filter.py` emits after cutting the spans this robot cannot
reproduce. Slicing happens before `resample()`, so replaying a span is
bit-identical to that span inside the full episode rather than a re-timed
approximation of it. One implementation shared by both callers, here, next to
the frames it slices.

## umi_gripper.py

The TacCap/UMI **follower** gripper mounted on both arms. Shares no code with
`gripper.py`, deliberately — see below.

### It is not on the arm's CAN bus

`gripper.py` speaks DM4310 MIT frames through Marvin's end-effector CAN
pass-through, sharing the arm's `RobotConnection`. This one is a separate USB
serial link per side (CH343 → MCU → FDCAN → motor) and never touches the arm
SDK. Consequences worth knowing: the gripper works with the controller powered
down, a controller reboot doesn't disturb it, and there is no `wait_ready.py`
dependency.

The SDK is loaded from `TACCAP_SDK` (default
`~/Desktop/project/TacCap-Gripper/python`), the same env-var pattern
`MARVIN_SDK` uses — it is not a pip package.

### Why reusing `GripperThread` would have been wrong

Three reasons, in descending order of how quietly they'd hurt:

1. **Force limits are much smaller.** `UMI_GRIPPER_MAX_TAU` is 1.0 N·m
   against a ~0.35 N·m working grip. `GRIPPER_MAX_TAU` is 8.0,
   sized for a DM4310 whose vendor limit is 10.0. Carrying 8.0 across would
   mean *no* protection rather than a conservative one — the mechanism would
   fail before the threshold was approached.
2. **`GripperThread._run()` polls `motor.read_status()` every cycle.** The
   vendor docs are explicit that polling `GetMotorStatus` above ~100Hz stalls
   the firmware's own telemetry refresh. Feedback must come from
   `ControlLoop.observation()` instead.
3. **The 50Hz Python slew thread is strictly worse than what ships.**
   `ControlLoop` is a C++ background thread defaulting to
   `SubmitPhase.STREAM_LOCKED`: one command per received motor-status frame,
   landing in the ~9.86ms the MCU is known to be idle. The vendor measured
   6000 submits : 6000 frames : 0 missing that way, against 156–308 lost
   frames per run free-running at 100Hz. A Python timer thread is the second
   case. It also seeds its target at the measured position on `start()`,
   which is what `GripperThread.start()`'s manual enable → sleep → read →
   seed sequence was doing by hand.

What *does* carry over is the idea: force-limited position control, capping
how far the command may lead the measured position. `UMI_GRIPPER_GRIP_MARGIN`
is that cap, expressed in normalized units because that is what both the SDK
and the recordings speak. Grip force ≈ `kp × margin × stroke_rad`; the driver
prints the implied N·m at connect time so it is never a guess.

### Convention: 0 = closed

The SDK uses 0 = closed / 1 = open. So do UMI recordings. `gripper.py` uses
the inverse. This driver follows the SDK, so a recorded jaw value reaches
`set_target()` with **no conversion at all** — the conversion burden falls on
the older, less-used path instead of the main one. This is the payoff of both
ends being the same mechanism: with a PICO trigger driving an OmniGripper
there was no physical correspondence between finger travel and jaw opening,
and "closed" on one side did not mean "closed" on the other.

Two calibrations still differ in origin and are worth checking once against
each other if leader-side data ever looks offset: the leader normalizes by
encoder shaft angle from a manually-set zero (`EncoderMaxCal`), the follower
by motor radians from a close-to-stall auto-cal. Both are stroke fractions,
but "0" is set by a different procedure on each.

### Refusing to start

`connect()` rejects a gripper whose `GripperConfig` lacks the calibrated bit
(`flags & 0x0001`). Normalized position has no physical meaning without it and
the SDK throws rather than guessing, so failing loudly at startup beats
failing mid-motion.

Side resolution prefers a pinned firmware SN (`config.UMI_GRIPPER_SN`) over
the SDK's side rule. USB enumeration order can change; a gripper driven as the
wrong side is an expensive thing to discover on hardware.
