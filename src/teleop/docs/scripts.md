# scripts/ — standalone calibration and debug tools

These are outside the main control loop (`run_teleop.py`), each independently
runnable for setup, calibration, or diagnosing a specific failure mode.

## goto_joints.py

Point-to-point joint move. Used to recover an arm left in an odd pose after
teleop, or after a disable.

Safety design:
- Starts from the **measured** position, not a stale commanded one, to avoid
  a jump at enable.
- Cosine ease-in/ease-out (S-curve) interpolation, default peak 8°/s — well
  below `config.MAX_JOINT_RATE_DEG_S`. See `drivers.md`'s
  `send_joint_commands` / `move_to_joints` section for why this replaced the
  earlier constant-rate version.
- Per-frame check of commanded vs. measured deviation; aborts immediately if
  it exceeds `MAX_TRACKING_ERR_DEG`.
- Ctrl+C stops in place at any time.
- `ensure_clear()` (see `drivers.md`) runs before `prepare()` — retries+
  confirms error clearing instead of `prepare()`'s own fault check raising a
  bare traceback on a still-latched e-stop.

`--control-mode impedance` (`--impedance-type joint|cartesian`): same flags
and `ImpedanceConfig` yaml loading as `run_teleop.py`, factored into
`algos.solver_config.load_impedance_config()` so the CLI-value/yaml-type
mismatch guard can't drift between entry points. `move_to_joints` sends the
exact same S-curve reference regardless of mode (`send_joint_commands` calls
the same SDK function either way — position mode treats it as a hard
target, torque+impedance mode treats it as a compliant reference); driving
a *moving* reference through impedance mode this way is new as of
2026-08-15 and had, before this, only been exercised via `run_teleop.py`'s
teleop-follow path — test slow/single-arm/supervised the same as any other
first use of torque mode. Expect the tracking-error abort to trip more
readily than in position mode: compliance means the measured pose
legitimately lags/deviates from the reference more, that's the mode's whole
point, not evidence of a collision.

Hardware fact (verified on real hardware, 2026-08-13): when disabled, the arm
**brakes and holds its current pose** — it does not sag from gravity. An
earlier assumption ("no brake on power-off, it will sag") was wrong. This
makes `--release` safe to use directly; the default "stay enabled" mode is
only useful if you want to keep fine-tuning the pose afterward. The
`--release` help text and print output in the script have been corrected to
match.

Implementation notes:
- Interpolation, rate limiting, and tracking supervision all live in
  `drivers.arm_driver.move_to_joints`. `run_teleop`'s automatic homing uses
  the exact same function — don't duplicate the logic here, the two copies
  will drift apart.
- The post-move "hold" loop commands `q_end` (where the arm actually stopped),
  not `TARGET`. If the move was interrupted, holding `TARGET` instead would
  turn "stop where you are" into "snap to target."

## set_state.py

Switches one arm's control state in place (`position` / `impedance` /
`drag` / `disabled`) without commanding it to a new target — holds at
wherever it already is. For testing a mode switch in isolation (e.g.
feeling impedance compliance by hand) independent of `goto_joints.py`'s
point-to-point motion.

`--state drag` is a **different SDK feature from impedance**, not another
`cur_state` value — `Marvin_Robot.set_drag_space()`, layered on top of an
already-active impedance mode, for a human to physically move the arm by
hand (drag-teaching). Found by reading the vendor's own runnable demo,
`DEMO_PYTHON/showcase_joint_drag_arm_A.py` (python_doc_contrl.md's
interface docs list *two* APIs for this — `set_drag_space(dgType)` and a
newer `set_joint_drag`/`set_cart_drag`/`exit_drag` trio — the demo uses
`set_drag_space`, so that's the one treated as validated here, same
"real demo over docstring" rule as `docs/algos.md#impedanceconfig`'s
`rot_type`). Implementation:
- Enters torque+impedance mode (`set_state`+`set_impedance_type`).
- For **joint** drag only: explicitly sends low-stiffness `joint_k`/
  `joint_d` (`--drag-stiffness`/`--drag-damping`, default K=0.5 D=0.3)
  before `set_drag_space`. The vendor demo sends no K/D at all before
  dragging; a first live test (2026-08-15) that matched the demo exactly
  was springy and couldn't be dragged to a new position — most likely
  because the controller kept whatever K/D a *prior* `--state impedance`
  call had last set (this session's teleop-tuned K=5 from
  `configs/solver/impedance_joint.yaml`), since nothing resets it and the
  demo's own environment apparently didn't have that problem. Explicit low
  stiffness, grounded in python_doc_contrl.md's own guidance for
  teaching/collaborative-contact use ("需要极低刚度和中度阻尼"), overrides
  whatever was inherited. Cartesian drag (X/Y/Z/R) still matches the demo
  with no K/D call — untested, no evidence yet either way, and the
  Cartesian nullspace stiffness field has a hard 20–100 floor unlike
  joint stiffness's 0–22, so "just go low" doesn't transfer directly.
- `set_drag_space(dgType)` (1=joint, 2-4=Cartesian X/Y/Z, 5=Cartesian
  rotation), confirmed via readback (`inputs[idx].drag_sp_type`) before
  telling the operator it's safe to grab.
- While dragging: reference rate-limit-tracks the measured pose
  (`--drag-track-rate` deg/s/axis, default 15) — three iterations to get
  here, all from the same 2026-08-15 live-test session:
  1. Demo's own "read state, send nothing": `joint_cmd_pos` stayed at
     whatever was last commanded *before* drag even started (stale), so
     the low-K spring kept pulling back toward that one fixed point no
     matter where the arm was dragged to — "不管拖到哪都固定不住，弹回一
     个点".
  2. `q_cmd = q_measured` every tick, uncapped: fixed (1) — stays wherever
     released — but made `--drag-stiffness` irrelevant to gravity sag.
     Tracking error was ~0 *by construction* regardless of why the joint
     moved, gravity-caused drift included, so raising K to 10 gave the
     spring no error to act against and did nothing.
  3. Rate-limited to `--drag-track-rate`: real drift (gravity, or the arm
     outrunning the cap while being actively dragged) now accumulates a
     genuine tracking error for `K` to generate restoring torque from,
     while the reference still catches up and holds once motion stops —
     keeps (1)'s fix without reintroducing (2)'s. Lower rate → more
     resistant to drift, closer to the original fixed-reference behavior;
     higher → more free-floating. Needs empirical tuning per use, the
     default (15°/s) is a first guess, not a measured value.
- Exit is `set_drag_space(dgType=0)` before `disable()`, in a `finally` so
  it runs on Ctrl+C too — same "must exit drag before switching modes,
  else effects superimpose and get confusing" warning the SDK docs give
  for switching between drag spaces.

Two symptoms from the same live test are left as-is, pending more evidence
rather than more parameter changes (both may improve once `--drag-track-
rate` is tuned — gravity sag was reported before that existed — but
neither is confirmed fixed):
- One joint sagged under gravity even when not being touched — could still
  be explained by a uniform K across all 7 joints being lower than what
  that particular joint's gravity/load torque needs, independent of the
  tracking-rate issue above. A uniform low K trades "feels free everywhere"
  against "holds against gravity everywhere," and the joints carrying the
  most reflected weight lose that trade first. A non-uniform, per-joint K
  (higher on whichever joint sags, still low elsewhere) would be the next
  step if `--drag-track-rate` tuning doesn't resolve it, not implemented
  yet since it needs a specific joint identified by further testing.
- The wrist joints felt stiffer than the rest despite the same K/D —
  plausibly gearbox/mechanical friction on those joints (higher gear
  ratios are typically harder to back-drive) rather than anything the
  impedance law controls; if so, no K/D or tracking-rate value fixes it.
  Not confirmed either way yet.

**Gravity compensation (`--tool`, 2026-08-16, not yet hardware-verified):**
re-reading the SDK surfaced a likely root cause for the "one joint sagged"
symptom above that's more fundamental than K tuning. Torque/impedance
mode already runs the controller's own gravity-feedforward model, built
from the arm's own Mass/MCP/I in `robot.ini` (`fx_kine.py`'s load-config
comment) — that's *why* a low-K spring can hold a bare arm up at all, it
isn't fighting the arm's own weight, only whatever the model doesn't know
about. The mounted gripper's mass was never registered with the
controller, so its weight falls squarely in that gap: an unmodeled load
the feedforward can't cancel, felt as sag on whichever joint carries the
most of it. `Marvin_Robot.set_tool(arm, kineParams, dynamicParams)` is
the SDK's hook for this — `dynamicParams` is `[mass, com_x, com_y, com_z,
ixx, ixy, ixz, iyy, iyz, izz]` (kg / mm from the flange / inertia, per
`DEMO_PYTHON/showcase_set_save_tool.py`, inertia optional per that demo).

Tool parameters live in `configs/tool/<name>.yaml`
(`drivers.tool_config.ToolConfig`, same typed-YAML pattern as
`algos.solver_config.ImpedanceConfig`) instead of raw CLI numbers —
one file per mounted tool, checked in and maintained over time rather
than re-typed by whoever happens to be running drag mode that day.
`configs/tool/omnigripper.yaml` is the entry for `drivers/gripper.py`'s
OmniGripper (DM4310). Per-arm (`arms.A`/`arms.B` in the yaml,
`ToolArmConfig`): the gripper's mount isn't guaranteed symmetric between
the two arms, so each arm needs its own entry; `load_tool_config(name,
arm, repo_root)` fails loudly (listing which arms *are* covered) if the
requested arm has none, rather than falling back to the other arm's
numbers or to zero. `set_state.py --state drag/release --tool omnigripper`
loads and applies whichever arm `--arm` selected; omitting `--tool` leaves
tool params untouched. Doesn't replace the per-joint-K idea above if this
alone doesn't fully explain the symptom — the two aren't mutually
exclusive.

**Filling in real numbers (2026-08-16):** SDK's own identification
pipeline, `Marvin_Kine.identify_tool_dyn(robot_type, ipath)`
(`DEMO_PYTHON/showcase_identy_tool_dynamic_{SRS,CCS}_B.py` —
`robot_type=1`/CCS is ours, per `config.KINE_CFG` = `ccs_m6_40.MvKDCfg`;
SRS/`robot_type=2` is a different machine). Two phases:
- **Online collection** (needs the real arm): run the same PVT
  identification trajectory (`CommonConfig/LoadData_ccs/LoadData/
  IdenTraj/LoadIdenTraj_MarvinCCS_{Left,Right}.fmv`) twice — once with
  the tool mounted (→ `LoadData.csv`), once with it physically removed
  (→ `NoLoadData.csv`) — via the demo's `collect_identy_data()`. This is
  the part that needs hands-on hardware time (mount/unmount + watch a
  60s trajectory run, twice).
- **Offline identification** (no hardware): `identify_tool_dyn` diffs
  the two datasets and returns `[m, mcp_x, mcp_y, mcp_z, ixx, ixy, ixz,
  iyy, iyz, izz]` — directly `ToolArmConfig.dynamic_params()`'s shape.

Undocumented layout requirement, found by `strace`-ing the call rather
than trusting the demo/docstring (the printed error message itself names
the wrong filename — `LoadIdenCfg_MarvinCCS.txt`, no underscore — when
what it actually opens is `LoadIdenCfg_Marvin_CCS.txt`, underscore,
nested one level down):
```
<ipath>/LoadData.csv
<ipath>/NoLoadData.csv
<ipath>/CfgFile/LoadIdenCfg_Marvin_CCS.txt   # or _SRS_ for robot_type=2
```
`CfgFile/LoadIdenCfg_Marvin_{CCS,SRS}.txt` already ship under
`CommonConfig/LoadData_ccs/LoadData/CfgFile/` — copy (or point `ipath`
at a directory containing) that alongside whichever arm's
`LoadData.csv`/`NoLoadData.csv`.

Arm B (right) didn't need a live-hardware collection pass at all: the SDK
ships pre-collected omnigripper data at `CommonConfig/LoadData_ccs/
LoadData/omini-gripper-tool-parameter/right-arm/{LoadData,NoLoadData}
.csv` — someone already ran the online phase for this exact tool. Running
`identify_tool_dyn` against it offline (no robot connection) gave
`configs/tool/omnigripper.yaml`'s current `arms.B` entry — see that
file's `source:` field for the full provenance note, including an
unexplained mismatch between the call's printed debug summary and its
actual return value for the inertia terms. This is vendor reference-unit
data for the same DM4310 model, not collected against this project's own
physical robot — a reasonable estimate, not a guaranteed exact match.

Arm A (left) had no equivalent bundled dataset — only "right-arm" data
ships — so it needed its own online collection pass
(`LoadIdenTraj_MarvinCCS_Left.fmv`, `robot_id='A'`). Done (2026-08-17) via
`tool_calib_collect.py --arm A --tool omnigripper` +
`tool_calib_identify.py`, giving `configs/tool/omnigripper.yaml`'s
`arms.A` entry — measured against this project's own physical robot, not
vendor reference data like `arms.B`. The two arms' numbers noticeably
differ (mass ~1.5% lower on A, COM offset in a different direction),
confirming the mount isn't mirrored — good thing `arms.B`'s numbers were
never copied over as a stand-in.

**`--state release` (2026-08-16, not yet hardware-verified):** a second,
simpler way to get a gravity-compensated drag, found in the same
SDK read — `Marvin_Robot.set_state(state=4)`
(`STATE_COOP_RELEASE`/"协作释放", `DEMO_PYTHON/showcase_collaborative_
release.py`) is a distinct `cur_state` value, not `set_drag_space`
layered on impedance mode like `--state drag` above. No K/D, no
`set_impedance_type`, no drag-space call — the vendor demo's own words:
"有重力补偿,可以手轻松的扭/拽/拖机器人" (gravity-compensated, the arm can
be twisted/pulled/dragged by a light touch). Trade-off vs. `--state
drag`: it's all-axis zero-force, not directional — you can't restrict it
to one Cartesian axis or joint the way `--drag-space X` does. Exit is
just `set_state(state=0)`, same as `disable()`, so no `finally`-block
cleanup is needed the way `dgType=0` is for drag.

What "gravity-compensated" does and doesn't buy you here, since it's easy
to over-read: with no K/D at all, commanded torque *is* the gravity
feedforward, full stop — no spring pulling back toward a reference, so in
principle wherever the arm is let go is an equilibrium (zero net torque,
zero velocity → stays put), unlike `--state drag`'s reference-tracking
spring. But that equilibrium is only as good as the feedforward model:
`--state release` reads the *same* controller-side model as `--state
drag`, so an unregistered tool causes the exact same residual-torque drift
here, on the same joint — `--tool` isn't drag-specific, `release` needs it
too if the symptom above is really an unregistered-tool problem and not
something `--drag-track-rate` was masking.

Implementing both surfaced that `drv.engaged` was never set to `True` on
either the pre-existing drag path or the new release path — both call
`conn.robot.set_state()` directly instead of going through `drv.prepare()`
(which is what normally sets it). Left as `False`, the outer `finally:
drv.disable()` silently no-ops (`ArmDriver.disable`'s own guard), so on
exit the arm stayed energized in torque/CR mode instead of actually
disabling. Both branches now set `drv.engaged = True` once the state
switch is confirmed, so exit actually de-energizes the arm as the module
docstring already claimed it did.

This is the first time this project's software has driven `set_drag_space`
or `state=4` at all — elevated risk beyond the usual "first time torque
mode" case, since the entire point is a human's hand on an energized arm
at the same time it's being commanded (or gravity-compensated and free to
move on its own for `release`). Two-person operation recommended: one at
the arm, one at the e-stop.

`bin/set-state` is a thin wrapper (same repo-relative-path pattern as
`bin/get-current-pos`/`bin/go-home`) that maps its first two positional
args to `--arm`/`--state` and forwards everything else (`--impedance-type`,
`--drag-space`, `--drag-stiffness`, `--tool`, `--hz`, ...) unchanged,
so every `set_state.py` flag still works:

用法：
```
python3 scripts/set_state.py --arm A --state position
set-state A position
set-state A impedance --impedance-type cartesian
set-state B drag --drag-space X
set-state A drag --tool omnigripper
set-state A release --tool omnigripper
set-state A disabled
```

## gripper_calib.py

Steps the gripper through a list of target positions, pausing at each so a
human can visually confirm open/close direction and locate the mechanical
end-stops.

Why a separate script instead of reusing `tools/gripper_cycle.py`:
`gripper_cycle.py` does enable → move → disable on every call. Disabling
mid-sequence causes the gripper to spring back (~0.014 rad measured), which
corrupts the visual read of direction. This script stays enabled for the
entire sequence and only disables once, at the end.

Abort thresholds (carried over from `gripper_cycle.py`): stop immediately and
disable if torque exceeds `GRIPPER_MAX_TAU` (3.0 N·m) or position deviation
exceeds `GRIPPER_MAX_ERR_RAD` (0.5 rad). The `finally` block disables
unconditionally regardless of how the script exits.

## gripper_range.py

Measures gripper travel range: puts the motor in a zero-torque "free" state
and lets the operator move the jaw by hand through its full range while the
script logs position.

Principle: in MIT mode, torque = `kp*(q_des-q) + kd*(dq_des-dq) + tau`.
Sending `kp=0, kd=0, tau=0` makes commanded motor torque exactly zero, so the
gripper can be freely moved by hand. The DM4310 replies once per command
frame, so continuously sending this null command is what keeps a live
position readout coming back.

This is faster than probing end-stops step by step with `gripper_calib.py`,
and carries no risk of commanding into the mechanical limit.

## gripper_smoke.py

Smoke-tests only `GripperThread.start()` timing, without touching the arms.

Why isolated: `run_teleop.py --gripper` switches the arms into position mode
*before* starting the gripper thread. If the gripper then errors, the arms
are already powered — and before 2026-08-13, `disable()` didn't wait for
completion, which could leave a powered arm behind. Testing gripper startup
by itself cleanly localizes any problem to the gripper.

What it verifies: `gripper.py`'s startup sequence (enable → sleep → read
state). This is the exact fix point for a bug where the liveness probe
passed but `run_teleop` reported "no response."

Physical warning: `GripperThread`'s initial target is 0.0 (fully open); as
soon as the thread runs, it ramps toward fully open at `GRIPPER_SLEW_RAD_S`.
If the gripper is holding something, it will let go. By default this script
only calls `start()` without running the ramp thread (gripper doesn't move);
pass `--run <seconds>` to actually watch it move.

## net_check.py

XR link health check: frame rate, dropouts, button range. Run once before
getting on the real robot.

Interpretation thresholds (Session D measured baseline):

| max age    | verdict | note |
|------------|---------|------|
| < 0.15s    | good    | a dedicated 5GHz link measured 0.130s |
| < 0.25s    | usable  | `XR_STALE_S` itself is 0.25; beyond that the robot freezes |
| > 0.25s    | bad     | phone hotspot on 2.4GHz measured 0.233s, with occasional 13.4s dropouts |

Measured button range: grip full-scale = 1.000 on both hands.
`CLUTCH_THRESHOLD=0.5` is half of full scale.

## sine_probe.py

Small single-joint sine test for target/feedback phase lag. Uses the same
`send_joint_commands → set_joint_cmd_pose → send_cmd` path as teleop, with
no IK, model chunks or host low-pass filtering. `--execute` enables the
selected arm in position mode around its **current measured pose**. The other
six joints hold that pose. It does not home the robot or operate the grippers.
Default: arm A (left), J7, ±1 degree, 0.5 Hz, six full-amplitude cycles,
two-second amplitude ramps at each end, then one second holding the center
(17 seconds total). Normal completion disables the selected arm. Ctrl+C or
a fault stops/disables it where it is; it does not attempt a return motion.

```bash
# Offline preview: no SDK import, connection, or motion.
.venv/bin/python scripts/sine_probe.py

# Real motion: keep the chosen joint's sweep clear and use one command sender.
.venv/bin/python scripts/sine_probe.py --execute --arm A --joint 7 \
  --amp-deg 1 --freq-hz 0.5 --hz 250

# Separate comparison run at 100 Hz; retain the same pose and other settings.
.venv/bin/python scripts/sine_probe.py --execute --arm A --joint 7 \
  --amp-deg 1 --freq-hz 0.5 --hz 100

# Offline re-analysis of a saved run.
.venv/bin/python scripts/sine_probe.py --analyze bench/results/sine_TIMESTAMP

# Fake-transport and synthetic-phase tests; no hardware.
.venv/bin/python -m unittest bench.test_sine_probe -v
```

Send and read deadlines are separate, serviced by one SDK-owning loop.
`--read-hz` defaults to 1000 Hz **requested polling**; neither Python nor the
controller is assumed to achieve it. The script records actual host call
times, does not burst overdue commands, and saves only fresh feedback frame
serials. It prints selected-joint sent/target/fb_cmd/actual values every 0.5 s.
`--cycles` controls the bounded test duration.

Output goes to a new `bench/results/sine_TIMESTAMP` directory (ignored by
Git), or a new directory selected with `--output`:

| File | Contents |
|---|---|
| `sent.csv` | Host send begin/end nanoseconds and all seven commanded joint angles, degrees |
| `frames.csv` | Host read begin/end nanoseconds, frame/input serials, state/error, all seven axes of `joint_cmd_pos`, `fb_joint_cmd`, `fb_joint_pos` |
| `meta.json` | Wave settings, current-pose anchor, live-verified configured speed/acceleration ratios, controller version and exit status |
| `summary.json` | Fitted amplitudes, residuals, phase-derived lags and observed send/read intervals |
| `tracking.png` / `tracking.pdf` | Automatically generated two-curve position plot, raster and vector |

The plot shows two steady sine cycles: **Command** in dashed blue and
**Measured position** in solid orange, at their original host timestamps.
Both curves subtract the same starting joint angle for an uncluttered
displacement axis. The header reports the phase-derived lag and observed
command rate; the footer identifies the longer interval used for the fit.
The plot preserves the lag visually: it does not time-align the curves or
replace the measurements with fitted/smoothed sine waves. Both normal test
completion and `--analyze` generate the PNG/PDF without opening a GUI.

`joint_cmd_pos` is the SDK input-target readback; `fb_joint_cmd` is the SDK
output-command field. Neither field's name proves its internal firmware
processing stage. `fb_joint_pos` is achieved/measured joint position. The
analysis compares all three against the locally sent sine and compares
target readback directly against achieved. Positive lag means the response
trails the reference. A zero/very-small-amplitude signal (<0.01 degree)
produces a null lag, not a claimed measurement.

The phase fit excludes the ramps and the first full-amplitude settling cycle,
requires at least two remaining cycles, and fits sine/cosine plus a constant
offset at the requested frequency using actual host times. Phase is modulo
one period: a single sine cannot distinguish a delay from one differing by
whole periods. Inspect fitted amplitudes and residuals before interpreting
distorted/clipped motion as a fixed delay. Host read times are **not** encoder
capture times; frame serials are saved as identifiers and never silently
converted into a presumed 1 kHz clock. These records cannot alone isolate
network latency, controller receive time or the firmware servo period.

The probe uses existing `config.VEL_RATIO` / `ACC_RATIO`, verifies both on
readback, checks SDK joint limits plus config overrides/margins and validates
conservative waveform velocity/acceleration bounds. It refuses an existing
target/feedback gap >0.5 degree before enabling, stops at >3-degree tracking
error, >0.5-degree movement of a holding joint, or selected-joint travel
outside ±(amplitude+1 degree). Unchanged feedback for 100 ms and a send gap
exceeding max(50 ms, five configured periods) also abort. Parameters are
restricted to small tests (amplitude ≤3 degrees, sine ≤2 Hz). These joint
checks do not establish Cartesian/self-collision clearance. No automatic
fault clearing occurs. All Python exception paths after preparation attempt
stop/disable and preserve collected CSVs; an analysis failure after a short
run does not discard the raw recording.

## get_current_pos.py

Prints both arms' current joint positions (measured `q` and commanded
`q_cmd`, degrees) and `cur_state`/error code. Purely read-only: opens its
own `RobotConnection`, calls `ArmDriver.state()` (a `subscribe()`, not
`clear_error`/`prepare`/`disable`), and closes -- never sends a command, so
it's safe to run any time, including while another script or
`run_teleop.py` already has the arms engaged.

`bin/get-current-pos` is a thin wrapper around this script (resolves the
repo root from its own location, then execs `.venv/bin/python3
scripts/get_current_pos.py`) -- tracked in git, not machine-specific. Put
`bin/` on `PATH` once (e.g. `export PATH="/path/to/teleop/bin:$PATH"` in
`~/.bash_aliases`) and `get-current-pos` runs from any directory; any future
`bin/*` wrapper added the same way needs no further shell-config edits.

用法：
```
python3 scripts/get_current_pos.py            # config.ARMS 里配的臂都读
python3 scripts/get_current_pos.py --arms A
get-current-pos                                # bin/ 里的封装，等价于上面默认调用
```

## check_errors.py

Check-and-clear-faults, and nothing else. Per arm: read `cur_state`/
`err_code`, and if either says faulted, run `ensure_clear()` (see
`drivers.md`) until confirmed clean or `--retries` is exhausted. No
`prepare()`, no `set_state`, no `send_joint_commands` -- the servo is never
enabled and the arm never moves.

Why it exists as its own script: every other entry point
(`run_teleop.py`, `home_now.py`, `goto_joints.py`, `set_state.py`) already
calls `ensure_clear()`, but only as a prelude to *moving*, and each aborts
with a "先用 MarvinPlatform GUI 排查" `RuntimeError` if the clear fails.
After an e-stop -- `err_code=13` (Emcy) stays latched after the button is
twisted back out, and the servo cannot be re-enabled until it is dropped --
there was no way to just clear the fault and look at the result before
committing to a motion.

- Already-clean arms are reported and skipped, no pointless `clear_error()`
  fired at a healthy arm.
- `--arms` selects arms (defaults to `config.ARMS`); an arm that isn't
  `A`/`B` is rejected *before* the connection is opened, so a typo doesn't
  cost a connect/disconnect cycle.
- `--retries`/`--wait` pass straight through to `ensure_clear()`. The
  defaults (5 × 0.3 s) match what the other scripts use; raise them if a
  controller is slow to drop a bus fault.
- Exit code is 0 only if *every* requested arm ended clean, so it composes:
  `check-error && go-home`. A residual fault exits 1 with the same "清错清
  不掉，去 GUI 排查" pointer the other scripts give -- a fault that survives
  five `clear_error()` calls is a live condition (e-stop still pressed, bus
  or servo alarm), not a latch.
- Never engages, so the `finally` only closes the connection -- but it does
  have to close it, or port 4730 stays held against MarvinPlatform GUI (see
  `RobotConnection`'s own connect-failure message).

`bin/check-error` is a thin wrapper (same repo-relative-path pattern as
`bin/get-current-pos`) that turns a positional arm spec into `--arms`:

用法：
```
python3 scripts/check_errors.py               # config.ARMS 里配的臂都清
python3 scripts/check_errors.py --arms A
check-error                                    # 两臂都清（等价于不传 --arms）
check-error A                                  # 只清臂 A
check-error AB                                 # 显式两臂
```

## home_now.py

Emergency/standalone homing -- moves the configured arms to
`config.HOME_JOINTS` without going through XR/teleop follow at all. Same
slow-rate, abort-on-excess-tracking-error path as `core/homing.py`'s
end-of-run homing (constant `config.HOME_SPEED_DEG_S`), just reachable
without a full `run_teleop.py` session. `--arms` (e.g. `--arms A`, `--arms
AB`) selects which arms; defaults to `config.ARMS` (both).

`bin/go-home` is a thin wrapper (same repo-relative-path pattern as
`bin/get-current-pos`) that turns a positional arm spec into `--arms`:

用法：
```
python3 scripts/home_now.py                   # config.ARMS 里配的臂都归位
python3 scripts/home_now.py --arms A
go-home                                        # 两臂都归位（等价于不传 --arms）
go-home A                                      # 只归位臂 A
go-home AB                                     # 显式两臂都归位
```

## tool_calib_collect.py / tool_calib_identify.py

Together, the 3-step pipeline behind `configs/tool/<name>.yaml`'s
mass/COM/inertia numbers (see `set_state.py`'s gravity-compensation
section above and `drivers/tool_config.py`): 1. collect no-load PVT data,
2. collect load (tool mounted) PVT data, 3. offline-diff the two into
`[m, mcp_x, mcp_y, mcp_z, ixx, ixy, ixz, iyy, iyz, izz]`. Steps 1+2 are one
script (`tool_calib_collect.py`, needs the real arm); step 3 is a separate
script (`tool_calib_identify.py`, pure offline, no robot connection).

Both wrap the same SDK calls as `DEMO_PYTHON/showcase_identy_tool_dynamic_
CCS_B.py` (`collect_identy_data()` / `run_offline()`) instead of hand-
running that demo's "uncomment one block, run, re-comment, repeat" flow
across three separate launches.

`tool_calib_collect.py --arm <A|B> --tool <name>`: connects, then for each
of the two passes (no-load, load) — pauses for the operator to
mount/remove the tool, **moves the arm to all-zero joints** with a direct
point-to-point move (`--speed`, same primitive as `goto_joints.py --to
0,0,0,0,0,0,0` — deliberately *not* `core/homing.py`'s "home" concept:
`config.HOME_JOINTS`/`HOME_SPEED_DEG_S` are an unrelated parking pose and
speed that only share the English verb), a fixed, trivially-reproducible
reference pose so both passes start identically. Then switches to PVT mode
(`state=2`), uploads the arm's `LoadIdenTraj_MarvinCCS_{Left,Right}.fmv`
trajectory, collects ~60s of joint-position/torque data, and washes the
raw dump into the plain numeric CSV `identify_tool_dyn()` expects. Output
lands at
`data/tool_calib/<tool>/<arm>/{NoLoadData.csv, LoadData.csv,
CfgFile/LoadIdenCfg_Marvin_CCS.txt}` — the last one copied from the SDK's
own `CommonConfig/LoadData_ccs/LoadData/CfgFile/`, required alongside the
two CSVs per the undocumented layout requirement noted above. Same
elevated-risk category as `--state drag/release`: PVT mode drives the arm
under its own internal speed/accel, not this script's rate limiting —
watch the whole pass.

`tool_calib_identify.py --dir <that output dir>`: calls
`Marvin_Kine.identify_tool_dyn(robot_type=1, ipath=...)` against the two
CSVs and prints the result plus a ready-to-paste
`configs/tool/<name>.yaml` block (`--arm`/`--tool` only affect how that
block is formatted). Verified against the SDK-bundled omnigripper
right-arm reference data — reproduces `configs/tool/omnigripper.yaml`'s
existing `arms.B` entry exactly. Does **not** write the yaml file itself:
that file is hand-curated with a prose `source:` note per entry, and a
plain YAML dump would silently strip every comment in it — paste the
block in by hand and fill in `source:` with the real collection
date/notes, same "measured or identified, never guessed" rule as
`drivers/tool_config.py`.

用法：
```
python3 scripts/tool_calib_collect.py --arm B --tool omnigripper
python3 scripts/tool_calib_identify.py --dir data/tool_calib/omnigripper/B --arm B --tool omnigripper
```

## umi_replay.py

Replays a recorded UMI episode on the real robot — both arms and both
follower grippers from one frame stream and one control loop.

Structurally it is `run_teleop.py` with the live XR source swapped for a
recorded one. Same `ArmChannel` pipeline, same `ControlLoop`, same
homing-before-disable discipline. Only the source differs, which is the point:
if the replay behaved differently from teleop, the replay would be testing
something other than what runs live.

`ArmChannel` gained an `axis_map=` parameter for this, defaulting to
`config.AXIS_MAP` — the input device decides the source frame, not the robot,
so a non-PICO source passes its own table instead of mutating `config`. It
later gained `tool=` for the same reason: a recorded UMI pose is a *fingertip*
midpoint, so this path retargets the fingertip while live PICO teleop keeps
retargeting the flange (`algos.md#tool_framepy`). The pre-flight preview
tracks both points and checks the keep-out box against each — with a tool
frame set they are tens of mm apart, and the tool reaches the body first.

`--frames A:B` replays one span in recorded-frame indices, which is how the
output of `scripts/umi_filter.py` feeds back in here.

### `--solver` defaults to `diff` here, unlike `run_teleop.py`

This is the one place the two entry points deliberately disagree.
`run_teleop.py` defaults to the analytic solver (`ik_solver.ArmIK`);
`umi_replay.py` defaults to diff-IK (`algos/diff_ik.py`,
`algos.md#diff_ikpy`), configured from `configs/solver/diff.yaml`.
`--solver analytic` selects the old behaviour, `--solver-config` points at
a different tuning.

Why the split: live teleop has a human in the loop who watches the tool and
corrects continuously, so a solver that occasionally rejects a frame costs
nothing but a hesitation. A recorded replay has nobody in the loop — a
rejected frame is a frame of the demonstration that simply does not happen,
and the arm holds its previous command while the recording moves on.
diff-IK is a rate controller: it always takes a bounded step *toward* the
target rather than one-shot solving for it, so tracking lag replaces
outright rejection, and its own velocity box keeps per-frame steps inside
the budget the safety gate would otherwise have to clamp. On
`replay_test` this shows up as 0 clamped frames (arm A, left hand) where
the analytic solver needed 15.

Two consequences worth knowing:

* **The nullspace controller is off** under `--solver diff` — `DiffIKSolver`
  doesn't support `NullSpaceController`'s reentrant probe solves. Redundancy
  handling comes from the QP's own `mu_nullspace` cost term instead
  (`1000.0` in the shipped YAML). The channel prints this at construction.
* **The pre-flight preview uses the same solver as the run.** It has to:
  diff-IK and the analytic solver pick different elbow configurations for the
  same target, so a preview solved by the other one is a preview of a
  different trajectory — and the envelope it prints is the only thing in this
  stack that can catch a trajectory heading into the robot body.

`replay.py --umi` (stage 1 below) defaults to `diff` for the same reason, so
the offline rehearsal and the hardware run agree by default. Both entry
points build the solver through `algos.diff_ik.build_from_config()`, so
their tolerances and margins cannot drift apart.

### The three-stage ladder

| stage | command | what it proves |
|---|---|---|
| 1 | `replay.py --umi DIR --hand left` | pipeline + coordinate frame, **no robot at all** |
| 2 | `scripts/umi_replay.py DIR --dry-run` | connection, `prepare()`, live tracking error; sends nothing |
| 3 | `scripts/umi_replay.py DIR --speed 0.3` | real motion, slowly |
| 4 | `scripts/umi_replay.py DIR --gripper` | + jaws |

Stage 1 is the only one that runs without hardware: `RobotConnection` always
connects, `dry_run` only suppresses sending. Do not skip it — it is where a
wrong `AXIS_MAP_UMI` shows up as an IK rejection rate rather than as motion in
the wrong direction.

### `--speed` divides, it does not multiply

`resample(raw, CONTROL_HZ / speed)` then playing back at `CONTROL_HZ` is what
makes `--speed` a pure time rescale: `0.5` asks for twice as many interpolated
frames, which take twice as long to emit — identical geometric path, half the
Cartesian velocity. Multiplying instead (the obvious-looking version) makes
`--speed 0.3` play three times *faster*, which is exactly backwards from what
a cautious first run wants.

### Measured on `replay_test` (299 frames, 9.93s, both arms)

Offline, `SCALE=1.0`, both arms: **100% IK success, 100% safety-gate pass**.
The trajectory has one brief fast moment at t≈3.3s (recorded frame ~100) that
exceeds the joint-rate budget by ~17% — peak 0.42°/frame against 0.356°
allowed at `VEL_RATIO=55`. The gate clamps it for ~34ms, which is what it is
for; the resulting lag is ~0.03°.

| `VEL_RATIO` | arm A clamped | arm B clamped |
|---|---|---|
| 55 (default) | 8 frames | 13 frames |
| 60 | 6 | 2 |
| 65 | 0 | 0 |

`--vel-ratio 65` clears it on both arms. Raise it in verified steps, not in
one jump — see `config.md#speed-budget`.

Cartesian peak in the recording is 463 mm/s (left) against a 446 mm/s budget
at `VEL_RATIO=55`, and rotation peaks at 106°/s against 107°/s. This demo sits
right on the speed budget; a livelier one will not.

Counterintuitively `--scale 0.5` clamps *more* than `--scale 1.0` (15 vs 8
frames on arm A). That is not a bug: a smaller scale is a different path
through joint space, and this one passes nearer a configuration where the same
Cartesian motion costs more joint motion. Mean per-frame joint step does drop
as expected (0.062° vs 0.070°).

### 事故记录：2026-08-25 A 臂险些撞本体

第一次实机运行（`--arms A --speed 0.3 --scale 1.0`，从 HOME 起步）跑了约 1.5 秒后
操作员拍下急停。复盘结论：

**不是速度问题。** 峰值指令关节速度 30°/s（额度 89），笛卡尔滞后 0.3mm。日志里那个
5.04° 跟踪误差和 39.9mm 滞后是**急停之后**的产物 —— 手臂冻结、指令继续发 0.5 秒，
误差累积到 sustained 阈值。时间线吻合（急停 → 约 125 帧 → 触发）。安全链路工作正常。

**是方向问题。** 这段 demo 是"双手内收"动作：左手从 world y=+0.251 扫到 +0.098，
内收 153mm。映射到 A 臂基座系（X 前 / Y 下 / **Z 左**）就是往 −Z 扫 153mm。而 A 臂在
HOME 时 TCP 距中心线只有 174.5mm：

| 起始构型 | 起点 TCP Z | 轨迹最小 Z | 峰值关节速度 |
|---|---|---|---|
| HOME | +174.5 | **+22.2** | 89.1°/s（贴限幅） |
| 事故中止位置 | +120.1 | **−32.3**（越过中心线） | 89.1°/s |
| `HOME_WAYPOINTS['A']` | +476.4 | +324.1 | 83.6°/s |

三者 IK 全部 2484/2484 通过 —— 因为**这套栈里没有任何自碰撞模型**。安全闸查的是关节
限位、关节速率、跟踪误差，没有一项能看见 TCP 正朝本体去。

**根因：相对映射保留的是位移，不是余量。** 示教者双手张开在身前起始，内侧空间充足；
机械臂从哪个构型起步决定了它有多少余量，而回放对此一无所知。

**修复**：`umi_replay.py` 在任何下发之前强制预演整条轨迹（用真实的 `ArmChannel`，
只是不发送），打印 TCP 在基座系的包络，并对照 `config.UMI_REPLAY_KEEPOUT` 判断；
再要求交互确认。`--no-preview` / `--yes` 可跳过，但不建议。

顺带修的第二个问题：原本用 `conn.check_and_clear_errors()`，它只 fire 一次
`clear_error()` 且不确认 —— `ensure_clear` 的 docstring 早就写明急停后这样不够，
`err=13` 会保持闩锁。已改用 `ensure_clear`。

## umi_filter.py

The embodiment feasibility filter's CLI: walk a recorded UMI episode through
the real pipeline, label every control period, print the spans this robot can
reproduce and cut the rest. Offline — no robot, no network, nothing sent.
Design rationale and per-frame codes: `algos.md#embodimentpy`.

```bash
python3 scripts/umi_filter.py ~/Downloads/replay_test
python3 scripts/umi_filter.py ~/Downloads/replay_test --arms A --speed 0.5
python3 scripts/umi_filter.py ~/Downloads/replay_test --sweep --out mask.json
```

Output ends in ranges that paste straight into a replay:

```
可用片段（录制帧下标，可直接喂给 --frames）：
  --frames 33:52      # 1.10..1.73s, 19 个录制帧
```

### Why recorded frame indices, not control periods

`--frames A:B` on `replay.py --umi` and `scripts/umi_replay.py` slices the
raw 30Hz stream *before* `resample()`, so replaying a span is bit-identical
to that span inside the full episode, and the indices mean the same thing to
a LeRobot dataset consumer as they do here. `drivers.umi_source.slice_frames`
is the one implementation both share.

### What is a verdict actually conditional on

The start configuration (`--start umi|home`, `--start-joints`), the tool frame
(`--tool` / `--tip`), the replay speed (`--speed`), the motion scale
(`--scale`) and the velocity gear (`--vel-ratio`). All of them go into the
JSON's `settings` block, because a mask without them is not reproducible.
Feasibility is not a property of the demo.

### `--recheck` (default on)

The main scan is one continuous take: span 3 starts wherever span 2 left the
arm. That is right for replaying the episode straight through and wrong for
using a span on its own — then the arm goes to the start pose first. So every
kept span is re-run from `q0` alone, and one that only worked because the arm
happened to be somewhere convenient is demoted. One extra IK pass per span;
the whole scan of `replay_test` takes ~1.3s per arm.

### `--sweep`

Same trajectory, several replay speeds, one table — the direct answer to "is
this demo infeasible, or just too fast?"

```
速度     臂A 可跟随 / 峰值速率  臂B 可跟随 / 峰值速率
1.00   100.00%  1.42×  100.00%  0.86×
0.70   100.00%  1.00×  100.00%  0.60×
0.50   100.00%  0.72×  100.00%  0.43×
```

Those ratios independently reproduce the on-robot numbers in
`umi与天机replay.md` §4.3 (arm A needs ~127°/s against 89°/s and got 10 frames
clamped; arm B peaked at 76.6°/s and got none).

### Exit code

Non-zero when nothing survives, so it can gate a scripted curation pass.

## tip_calib.py

Pivot-calibrates the flange → fingertip offset that
`algos.tool_frame.ToolFrame` needs, and prints the two lines to paste into
`configs/tool/<tool>.yaml` (`kine_offset` + `kine_offset_source`). Why the
offset matters at all: `algos.md#tool_framepy`.

**Read-only on the robot.** It calls `ArmDriver.state()` and nothing else —
no `prepare()`, no mode switch, no command, ever. Same category as
`get_current_pos.py`, and it is meant to run in a second terminal next to the
`release` session that lets you move the arm by hand:

```bash
# terminal 1 -- zero-force free-drive
set-state A release --tool umi

# terminal 2
tip-calib --arm A                       # or python3 scripts/tip_calib.py --arm A
python3 scripts/tip_calib.py --arm A --solve data/tip_calib/umi/A/samples.json
python3 scripts/tip_calib.py --self-test
```

### The procedure

Close the jaws fully, rest the two-finger midpoint on one fixed pointed
reference, and sweep the wrist through as many orientations as the arm allows
without letting that point move — one Enter per pose. Symmetric jaws keep the
midpoint still as they open, so a closed-jaw measurement is valid at every
opening (the same property `ee_transform.py` relies on for the leader).

Samples land in `data/tip_calib/<tool>/<arm>/samples.json`, so the fit can be
re-run later without the robot.

### Why the residual is not the quality metric

The fit is only as conditioned as the **orientation spread**. With the wrist
barely moving, `t` is unconstrained along the line of sight and the
least-squares solution is wrong *with a small residual* — the self-test
demonstrates exactly that: 8° of spread, 0.37mm residual, 2.9mm error. So the
gates are ≥4 samples, ≥60° of spread, ≤2mm residual, and the script refuses
to print a pasteable snippet until all three pass.

### What it cannot recover

The tool's *rotation*: a touched point is rotationally symmetric. That is
fine here — under relative mapping the tool rotation does not change a single
commanded pose (proved and self-tested in `algos/tool_frame.py`) — so
`kine_offset`'s a/b/c stay zero. They only matter if the same entry is later
handed to `Marvin_Robot.set_tool()` for controller-side Cartesian targeting.

It also cannot settle *where along the finger* the recorded TCP sits; the CAD
"EE frame" is the authority for that, and a few mm of axial disagreement can
survive this calibration.
