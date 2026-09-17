# core/ — orchestration layer design notes

Background and empirical data behind the terse in-code comments in
`core/arm_channel.py`, `core/homing.py`, `run_teleop.py`, and `replay.py`.

## arm_channel.py

### ArmChannel

One arm's full pipeline: controller pose -> retarget -> IK -> safety gate -> driver.
Extracted out of `run_teleop.py` so the entry point stays thin. Alternate
controllers (differential IK today, WBC/impedance control later) get
swapped in via the `solver` constructor argument — `run_teleop.py` only
needed one new CLI flag (`--solver`) to expose that, not a rewrite;
"shouldn't need to change" turned out to mean "one flag," not "zero
changes."

- Joint limits are sourced from `ArmIK` exactly once (`self.ik.lim_n/lim_p`)
  and handed to both the safety gate and the nullspace controller, so the two
  can never disagree about where the limits are.
- `track_err` / `track_err_max`: the core P0-2 health metric — how far the
  commanded joint position has drifted ahead of the measured one. In a
  healthy run this stays flat at a small value through the whole session; if
  it climbs monotonically toward `MAX_TRACKING_ERR_DEG` instead, the command
  stream is outrunning the controller.
- `_solve_joints` (nullspace probe callback): runs inside `self.ik.probing()`
  so speculative solves used only to test a candidate arm angle don't pollute
  the frame's real IK statistics — they're trial solves, not the frame's
  actual decision.
- `tool=` (an `algos.tool_frame.ToolFrame`) selects *which point* the channel
  retargets: `None`, the default, is the flange and is what live PICO teleop
  uses; the UMI paths pass the mounted gripper's fingertip offset, because a
  recorded UMI pose is a fingertip midpoint (`docs/algos.md#tool_framepy`).
  `ch.tip(q)` is the same point for anything that needs to ask where the tool
  actually is — the pre-flight envelope and the keep-out check both do, since
  the tool reaches the robot's body before the flange does.
- In `step()`, a `tracking_error` gate rejection is treated as backpressure,
  not necessarily a fault: an instantaneous overshoot just means this frame
  isn't sent and `q_cmd` doesn't advance, letting the servo catch up on its
  own without interrupting the operator. Only a *sustained* overshoot (or any
  `joint_limit` rejection) triggers a real fault and disengages the channel.

### `solver='diff'` wiring

`self.ik` (`ArmIK`) is constructed **regardless** of `solver` — it still
supplies `fk_xyzabc()`/`nsp_dir()` to the retargeter and `lim_n`/`lim_p`
to the safety gate either way, and `diff_ik.py`'s own docs are explicit
that this coordinate-format usage (not solving) is fine to depend on.
Only the actual per-frame solve call in `step()` branches on `solver`.

`solver='diff'` forces `self.ns = None` even if `cfg.NULLSPACE_ENABLED`
is set (with a printed warning) — `DiffIKSolver` doesn't support
`NullSpaceController`'s reentrant probe solves (see `docs/algos.md
#diff_ikpy`). Its own redundancy handling (`mu_nullspace`, a QP cost
term) is a different mechanism, configured independently via
`diff_ik_config`, not via `cfg.NULLSPACE_ENABLED`.

`diff_ik_config` must be provided (raises `ValueError` otherwise) when
`solver='diff'` — `ArmChannel` deliberately does not know how to load or
validate YAML itself (no `algos.solver_config` import here); that's
`run_teleop.py`'s job, so this module's only new dependency is
`algos.diff_ik`/`algos.kinematics`, not `pydantic`/`yaml`.

`diff_solve_ms_max` tracks the worst per-frame solve time for the whole
run, printed in `run_teleop.py`'s final summary — the number to compare
directly against `bench/compare_ik.py`'s offline solve-time percentiles
when checking whether a live dry-run's timing matches the offline
prediction (see `run_teleop.py` section below).

## homing.py

### home_arms

Moves each arm back to `config.HOME_JOINTS` after a run so the next session
starts from a known configuration.

- Must be called **before** the servos are disabled — an unpowered motor
  can't be moved. That's why it's wired into `ControlLoop`'s
  `before_disable` hook rather than `main()`'s `finally` block.
- Once homed it's safe to disable immediately: this arm **holds its current
  pose** when disabled rather than sagging under gravity (confirmed on
  hardware 2026-08-13). So homing once is homing for good until the next run.
- Why this runs every time, not just occasionally: the starting joint
  configuration directly bounds the reachable workspace. In Session D, a run
  that started from a config with J4 = -7.2° left only 4 mm of reachable
  space at the end-effector — 5 of 6 probe directions (+X/+Y/+Z all showed
  0 mm) were immediately unreachable, so pressing the clutch produced a
  screen full of `ik_failed`. The home configuration has 150-380 mm of
  headroom in all six directions.
- The mode guard checks the hardcoded `STATE_POSITION`, not
  `config.ARM_STATE` — found while wiring `--control-mode impedance`
  (which sets `config.ARM_STATE=STATE_TORQUE` for the whole run). The
  guard's actual intent, per its own comment, is "don't command motion on
  a ... non-position-mode arm"; checking it against `config.ARM_STATE`
  would have silently flipped that to "don't command motion unless
  *still* in torque mode" for an impedance run — i.e. it would have
  attempted `move_to_joints`'s rate-limited position-mode interpolation
  while the arm was still in torque mode, untested. Consequence now:
  after an impedance run, auto-homing is correctly skipped (arm stays in
  torque mode, `st['cur']!=STATE_POSITION`); use `scripts/home_now.py`
  separately afterward (its own fresh `config` import defaults to
  `ARM_STATE=1`, so it's unaffected by what the prior run set).

## run_teleop.py

### Thread model (deliberately decoupled — do not collapse into one loop)

| Thread | Rate | Job |
|---|---|---|
| SDK callback | 90 Hz | receive XR data, write into a slot (inside `xr_source`) |
| Main control loop | 250 Hz | read latest frame -> retarget -> IK -> safety gate -> send |
| Gripper | 50 Hz | slew-rate limiting + protection (`gripper.GripperThread`) |

XR packet loss must not stall the control loop, so the control thread always
reads "whatever frame is currently available" rather than blocking for a new
one. If no new frame has arrived within `XR_STALE_S`, a watchdog freezes
motion — the loop never blocks waiting for XR.

### CALIBRATED hard gate

`config.CALIBRATED = False` refuses to leave dry-run. Running uncalibrated
means the robot moves in a direction the operator didn't expect — the
easiest way to cause an incident — so this is enforced as a hard gate rather
than a warning.

### `--vel-ratio` / `--acc-ratio` handling

Goes through `config.apply_vel_ratio(...)` instead of a direct assignment
because three derived quantities (per-joint rate limit, nominal per-frame
step budget, Cartesian speed limit) must be recomputed together — setting
`VEL_RATIO` alone would leave the bottleneck between them inverted.

### Gripper thread vs. homing

The gripper thread and the arm joint commands share a single UDP channel
(sends that don't get through are retried for up to 30 ms). The gripper
thread is stopped before `home_arms()` runs so it doesn't contend with the
homing traffic.

### Vel-ratio readback check

`set_vel_acc` is only sent once, inside `prepare()`, but every joint command
frame calls `clear_set()`. The periodic status line reads back the
controller's *actually active* velocity percentage and flags it if it
disagrees with `config.VEL_RATIO` — that's the way to confirm `clear_set()`
hasn't silently reverted it.

### `--solver diff` gate (removed, commit 48d0816)

`--solver diff` used to refuse to run at all without `--dry-run` — see
`docs/algos.md#diff_ikpy` for the open gaps that gate existed to force a
second look at (no retry/backoff the way `ArmIK.solve_with_backoff` has,
unvalidated `W`/`lambda` weights, one unexplained 13.5ms solve-time
outlier before the J6/J7 fix that didn't reproduce afterward but was
never root-caused). Removed after a live `--dry-run --solver diff`
session's stats were compared against `bench/compare_ik.py`'s offline
numbers on the same kind of trajectory and found consistent, and
subsequent live-hardware validation (separate session) confirmed diff-IK
tracking as good as or better than the analytic solver — `--solver diff`
now runs live the same as `analytic`, no flag needed.

`--control-mode impedance` (below) has no equivalent gate to remove in
the first place — see its own section for why the same "dry-run then
compare offline numbers" pattern doesn't apply to it.

`--solver-config` defaults to `configs/solver/<solver>.yaml` (so
`--solver diff` alone picks up `configs/solver/diff.yaml`); pass a
different path to try alternate tunings without editing the checked-in
default. Both a missing file and a validation failure (typo'd key,
out-of-range value — see `docs/algos.md#solver_configpy`) print a clear
message and exit before any hardware is touched, same as the gate above.

The final per-arm summary reports `diff_solver.stats` (reject-reason
counts, same vocabulary `bench/compare_ik.py` uses) and
`diff_solve_ms_max` instead of `ArmIK`'s backoff/projection counts when
`solver='diff'` — those would just read zero, since `self.ik` is
constructed but never actually asked to solve in that mode (see
`arm_channel.py` section above).

### `--control-mode impedance`

Orthogonal to `--solver` — see `core/arm_channel.py`'s `ArmChannel`
docstring. `--solver` picks how `q` is computed; `--control-mode` picks
how the driver executes whatever `q` comes out (position vs. torque +
impedance). The two combine freely, e.g. `--solver diff --control-mode
impedance`.

Sets `config.ARM_STATE = STATE_TORQUE` at runtime (same "CLI flag mutates
the shared config module" pattern `--vel-ratio`/`--gripper` already use),
and loads/validates an `algos.solver_config.ImpedanceConfig` from
`--impedance-config` the same way `--solver-config` loads `DiffIKConfig`
— same `_load_typed_config` helper, same missing-file/validation-error
handling. `--impedance-type {joint,cartesian}` (default `joint`) picks
which of `configs/solver/impedance_joint.yaml` /
`impedance_cartesian.yaml` `--impedance-config` defaults to when not
given explicitly, and is cross-checked against the loaded file's own
`type` field — a mismatch (e.g. `--impedance-type cartesian` pointed via
`--impedance-config` at a file whose `type` still says joint) refuses to
start rather than silently running whichever one the file actually says.

**No dry-run gate, unlike `--solver diff`'s (now-removed) one — and
deliberately so, not an oversight.** `--dry-run` can't validate anything
about impedance control: `ArmDriver.prepare()`'s torque-mode branch is
*never reached* under `--dry-run` (that function returns before it, same
early-return that already skips `set_state`/`set_vel_acc`) — see
`docs/drivers.md`'s `ArmDriver.prepare()` section. So a clean `--dry-run
--control-mode impedance` run proves only that the new CLI flags/YAML
loading work, nothing about the SDK calls themselves. Unlike `diff_ik`,
there's also no offline equivalent to `bench/compare_ik.py` to fall back
on — impedance control's entire value proposition is a physical response
to real contact/gravity/inertia, which can't be exercised without
hardware. So first validation is necessarily a live run, not a staged
dry-run-then-compare step; see `docs/algos.md#impedanceconfig` for the
reasoning behind starting with joint impedance (`type=1`) rather than
Cartesian for that first live run.

`ArmDriver.prepare()` prints a one-time banner right before the mode
switch when entering torque mode, noting this is the first time the
control path is software-driven — not a gate, just makes sure whoever's
running it live sees it in the terminal, not just in a doc.

### Why `--dry-run --solver X` can't validate `tracking_error`/闭锁 — and the fix

`q_meas` in `step()` (`state['outputs'][idx]['fb_joint_pos']`, read via
`conn.subscribe()`) is **always** real hardware feedback, dry-run or not —
`RobotConnection.dry_run` only skips `send_cmds()` and the mode switch
(`drivers/arm_driver.py`). But `ArmChannel.q_cmd` is updated unconditionally
every accepted frame (`core/arm_channel.py`'s `step()`: `self.q_cmd =
v.joints` once the gate passes) regardless of whether anything was actually
sent.

So under `--dry-run` the real robot never moves — no command reaches it —
which means `q_meas` stays frozen at whatever pose it was in when the run
started, while `q_cmd` keeps advancing every frame the operator moves the
controller. `track_err = |q_cmd - q_meas|` stops measuring "is the servo
keeping up with the command stream" (what it means on a live run) and
instead measures "how far has the operator moved the controller since the
run started" — monotonic, with no recovery. It crosses
`MAX_TRACKING_ERR_DEG` almost immediately, and because it never comes back
down, `SafetyGate`'s `track_err_s` timer (`algos/safety.py`) always reaches
`max_tracking_err_s` (~0.5s default) and reports `sustained=True`, so
`ArmChannel` calls `rt.disengage(fault=True)` — a latch, every dry-run
session, regardless of solver correctness. That's "为什么 dry-run 秒闭锁":
structural to the dry-run + real-feedback combination, not a `diff_ik` bug,
and it will hit `impedance`/`WBC` the same way once they're registered in
`run_teleop.py`'s `_SOLVER_CONFIG_MODELS` and run through the same
`--dry-run --solver X` flow.

`replay.py` sidesteps the same problem a different, also-incomplete way:
`q = list(q_cmd)` right after each accepted step (`replay.py:171`, "replay
assumes perfect servo tracking") feeds the just-solved command back as the
next iteration's "measurement," so there's no real divergence to trip on.
That avoids the false latch but means `replay.py` can't validate
tracking-error dynamics either — it just fails in the opposite direction
(never trips, instead of always tripping). Between the two, there is
currently no offline path that produces a meaningful `track_err`/闭锁
number — which is exactly why the `--solver diff` gate message only asks
for reject-rate/jitter/solve-time to be compared against
`bench/compare_ik.py`, never tracking error or fault count (see
`run_teleop.py` section above). Don't apply the P0-2 criterion below to a
`--dry-run --solver X` session's fault-latch count — it will always be
nonzero there and that's not a signal of anything.

**The fix**, when a new solver's `track_err`/闭锁 behavior actually needs
validating before it's trusted on hardware: give dry-run a simulated
`q_meas` instead of real hardware feedback — a rate-limited follower toward
`q_cmd`, bounded by the same `cfg.MAX_JOINT_RATE_DEG_S` a real servo is
limited to, updated once per control period. Wire it in at the
`state['outputs'][idx]['fb_joint_pos']` read in `run_teleop.py`'s `step()`,
gated on `dry` (keep real feedback whenever a run isn't dry-run).
Deliberately not `replay.py`'s "perfect tracking" (that would trivially
never trip) — a rate-limited model actually lags behind fast commands the
way a real servo does, so `track_err` grows only when the command stream
genuinely outruns what the servo can deliver, and a resulting latch means
something again. This is generic to `solver`, so it's the one fix that
makes `--dry-run --solver X` tracking-error validation work for `diff` now
and `impedance`/`WBC` later, without ever touching hardware.

### P0-2 acceptance criterion

At the end of a run, the total fault-latch count across all arms must be
zero. If it's zero, the current `VEL_RATIO` tier is clean and it's safe to
step up (higher vel-ratio, or larger `scale`). If it's nonzero, check the
peak tracking error printed per arm: close to the limit means the command
stream is still outrunning the controller (lower `CMD_RATE_MARGIN` or
`scale`); far below the limit means something else tripped the gate (check
`ik_failed` counts in the IK stats). This assumes a live run against real
hardware — see the dry-run caveat above for why a `--dry-run --solver X`
session's fault-latch count doesn't carry the same meaning.

## replay.py

### Why this is the most valuable debugging tool in the project

It runs a recorded XR trajectory through the full pipeline — calibration
correctness, IK success rate, repeated rejects near the edge of the
workspace, per-frame step overspeed — all visible here, offline, at zero
robot risk. Every parameter change should be replayed before it's tried on
hardware.

### `synth_traj`

Deliberately shaped as "clutch pressed -> trace a circle -> release -> press
again" so the clutch state machine gets exercised too, not just the IK/
retarget math.

### `--force-clutch`

Bypasses the clutch state machine and forces engagement for the whole replay
— useful only for offline diagnosis of retarget/IK behavior on a recording
where the operator forgot to hold the clutch. It does **not** substitute for
validating the real clutch state machine.

### Starting joint configuration

`q = [44.04, -62.57, -8.92, -57.21, 1.45, -4.39, 2.1]` is the vendor demo's
"desk task" pose, used only because replay has no real robot to read a
current position from. On real hardware, use the measured current joint
angles instead.

### Zero-order-hold replay loop

Iterates at the **real control frequency**, not the recording's frame rate.
Recordings run at ~72 Hz while the control loop runs at 250 Hz — a 3.5x gap
in the per-step motion budget. Iterating directly over recorded frames would
overstate the per-frame joint step and understate the IK success rate,
making the resulting numbers unusable. The loop instead reproduces
`run_teleop`'s actual behavior: each control period reads whichever recorded
frame was most recently available at that timestamp.

### Verdict criteria (P2 acceptance, per the project plan doc)

IK success rate > 99% and the per-frame joint delta never exceeds
`cfg.MAX_JOINT_STEP_DEG`, for the whole replay.
