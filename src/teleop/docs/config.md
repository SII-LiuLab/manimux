# config.py — calibration history and tuning rationale

Single source of truth for every tunable in the teleop pipeline. This
file exists because `config.py` keeps only short, disambiguating tags
in-code; the "why" for each value — the incident, the measurement, the
session it came from — lives here instead.

Why `.py` and not YAML/JSON: no `pyyaml` in the target environment, and
JSON can't carry comments — most of this config's value is in the
rationale, so a commentable format was required.

## Hardware

- `HOME_JOINTS`: post-run parking pose. Starting configuration determines
  the usable workspace for the whole session. Session D measurement: arm A
  starting from `[39.2, -68.16, -85.47, -35.34, ...]` had its end-effector
  733 mm from the base against a 764 mm reachable radius in that direction
  — only 31 mm of margin, hitting the elbow singularity (J4→0) after just
  18 mm of +X travel, while -X had 541 mm (30:1 anisotropy). That's the
  root cause of "only one direction moves." Homing after every run avoids
  starting from a random (usually bad) pose. The two configured poses are
  half-flexed references for A/B with J4=-90°, far from the elbow
  singularity.
- `HOME_SPEED_DEG_S = 8.0`: an order of magnitude slower than teleop —
  homing isn't teleop, no need for speed.
- `KINE_CFG`: only trustworthy after the Session C FK verification passed.

## Coordinate calibration

Calibrated 2026-08-12: Session B measured the XR side, Session C measured
the base side, Session A composed them into `AXIS_MAP` below.
`run_teleop.py` allows real dispatch on `CALIBRATED = True`.

"Calibration complete" does not mean "direction is definitely correct":
the composition assumed a first-person same-orientation convention for
operator stance (see `AXIS_MAP` notes) that neither B nor C actually
measured. First real run must first confirm, at `scale=0.2` with
translation only, that manual motion direction matches robot motion
direction, axis by axis.

**XR_HANDEDNESS_FLIP_AXIS**: Session B measurement (2026-08-12, PICO 4U
hardware) — raw XR data is already right-handed (+X right / +Y up / +Z
back, X×Y=+Z holds directly), not the left-handed Unity convention
assumed at write time. `None` = no flip needed.

**XR_POS_TO_MM**: Session B confirmed units are meters → ×1000.

**XR_QUAT_ORDER**: Session B confirmed `'xyzw'` — fields are qx,qy,qz,qw,
and at rest qw≈1 with the other components →0, consistent with the
scalar component being last for a unit quaternion.

**AXIS_MAP** — right-handed XR → robot base frame axis mapping, composed
from B+C (2026-08-12).

Dict keys are XR's **basis vectors** (`xr_x`/`xr_y`/`xr_z`), not semantic
names. Reason: see `retarget.build_axis_matrix()` — PICO's +Z is
"backward," not "forward," so a key like `'xr_forward'` would silently
differ in sign from its value, which is undiagnosable in the field.

Raw measurements:
- Session B (XR side, already right-handed): +X=right +Y=up +Z=back (toward body)
- Session C (base side): +X=front +Y=down +Z=robot's own left

Composition needs one more fact that neither B nor C measured: **operator
orientation relative to the robot**. This uses the **first-person
same-orientation** convention — the operator "embodies" the robot, so the
person's front/up/right map to the robot's own front/up/right. This is
the standard teleop convention and matches this project's headset
stereo-view usage. If Session D finds left/right swapped (meaning the
actual stance is face-to-face), flip `xr_x` from `'-Z'` to `'+Z'`; leave
the other two lines alone.

Derivation (person's direction → robot's own direction → base axis):
- XR +X = person's right → robot's right → base -Z (+Z is robot's left)
- XR +Y = person's up → robot's up → base -Y (+Y is down)
- XR +Z = person's back → robot's back → base -X (+X is front)

All three signs negative, det = +1 (verified by `build_axis_matrix`, not
a reflection).

**Per-arm, not shared.** The two arms' local frames are not the same.
Evidence (read 2026-08-13; identical in both `ccs_m6_31` and the
`ccs_m6_40.MvKDCfg` now in use): gravity vector
`A = [0, +9.81, 0]`, `B = [0, -9.81, 0]`. Gravity is always "down" in the
world frame, so arm A's local +Y is down while arm B's local +Y is
**up** — the two arms' Y axes are inverted. Since DH parameters are
identical and both zero-poses put the end-effector at (0, 0, 870.5), this
is the same arm module mounted 180° rotated, not a mirrored part.
`ArmIK` solves poses in **each arm's own local frame**, and there is no
torso→shoulder transform anywhere in the config, so one `AXIS_MAP` cannot
be correct for both arms simultaneously.

Y-flip while staying right-handed ⇒ exactly one more axis must also flip;
which one (X or Z) depends on mounting orientation:
- 180° about X (Y,Z flip, X unchanged) → `{'xr_x':'+Z','xr_y':'+Y','xr_z':'-X'}`
- 180° about Z (X,Y flip, Z unchanged) → `{'xr_x':'-Z','xr_y':'+Y','xr_z':'+X'}`

`AXIS_MAP['A']`: ✅ measured, Session B+C (2026-08-12). Derivation above.

`AXIS_MAP['B']`: ⚠️ **not measured**. Derived from A's mapping plus the
gravity-vector Y-flip fact; the third axis takes the "180° about X"
branch, based only on operator-reported "lateral feels reversed," not a
measurement. Must verify before real use (scale 0.2, translation only,
confirm axis by axis):
- Up/down **must** be inverted — guaranteed by the gravity evidence. If
  measurement shows it is **not** inverted, the whole model is wrong;
  stop and re-derive, don't just flip signs.
- If front/back is normal, keep the current mapping; if front/back is
  also inverted, switch to `{'xr_x': '-Z', 'xr_y': '+Y', 'xr_z': '+X'}`.

## Retargeting

**SCALE = 0.5** (0.5 = 20 cm hand motion → 10 cm robot motion; raise
toward 1.0 once proficient).

Upper bound on `SCALE` is set by whether `hand travel × scale` stays
inside the arm's workspace. When `FOLLOW_ROTATION=False`, orientation is
locked, so pure translation drives the wrist (J6, limit ±60°) straight
into its limit. `traj_c.jsonl` measurement (65 cm hand travel):
- scale 0.5 → IK success 96.42%, 276 frames rejected for J6 entering the limit margin
- scale 0.35 → IK success 100%
- scale 0.2 → IK success 100%

Don't raise scale casually; enabling `FOLLOW_ROTATION` relaxes this
constraint somewhat.

**FOLLOW_ROTATION**: recommend `False` for first bring-up (translation
only, orientation locked), enable once stable.

**CLUTCH_THRESHOLD = 0.5**: originally 0.7, but `traj_b.jsonl` grip never
exceeded 0.675 in that recording (only one continuous >0.3 segment, 22
frames), so it's unclear whether 0.675 was an incomplete squeeze or the
PICO controller's actual max range — insufficient data either way. 0.5 is
clearly an intentional press with margin, avoiding a "clutch doesn't
respond" surprise on day one. Confirm actual full-scale value in Session D.

**ARM_ANGLE_LIMIT = 45.0**: originally 30, but measurement showed
automatic limit-avoidance needs up to 43° of arm-angle travel in the -Z
direction — capped at 30 it couldn't reach 48% of that need, raised to 45.

### Nullspace joint-limit avoidance (see `nullspace.py`)

The redundant DOF of a 7-axis arm — the elbow rotates about the
shoulder–wrist axis while the end-effector stays fixed — is used to pull
an axis approaching its limit back, transparently to the operator.

Measurement (arm A, home pose, orientation locked, 2026-08-13): -Z
direction reach improved 240→356 mm (+48%), overall +7%; the other five
directions showed zero engagement, zero change.

⚠️ `NULLSPACE_ACTIVATION_DEG` (the activation deadband) must not be
removed. An unconditional version keeps consuming arm-angle budget in
"fully extended" directions and actively hurts (+Z regressed 10%); with a
25° deadband, no direction regresses.

⚠️ Which axes the arm angle has leverage over is configuration-dependent
— it is not a general cure. Measured sensitivity at the home pose:
J2/J3/J5/J7 ≈ 0.7 °/° (recoverable), J1/J4/J6 ≈ 0 (not recoverable). J4
being exactly 0 is a geometric necessity — elbow flex angle is determined
solely by the shoulder-to-wrist distance, which arm-angle rotation does
not change. So nullspace avoidance is powerless when stuck on J4/J6 —
that's a genuine DOF shortage, not a tuning problem.

Cost: two extra IK solves per frame (± `NULLSPACE_PROBE_DEG` gradient
probe), ≈0.05 ms, negligible against a 4 ms period.

**NULLSPACE_PROBE_DEG = 0.5**: gradient sampling step. Must be small:
measured arm-angle sensitivity is 0.4–1.1 °joint/°arm-angle; a 2° probe
would move a joint by 2.2°, exceeding `IK_MAX_STEP_DEG` (1.8°), which
makes IK classify it as a branch jump and fall back to the AllJoint
solution (arm-angle independent) — all three probe points then return the
same solution and the gradient is silently erased, with the controller
showing zero engaged frames the whole run. 0.5° only moves joints ≈0.55°,
leaving margin. When this erasure does happen it is now counted in the
`n_blind` stat rather than failing silently.

**LOWPASS_HZ = 8.0**: controller position low-pass cutoff, set after
measuring rest-state noise in Session B. Too low adds latency, too high
doesn't filter jitter.

`CART_MAX_SPEED_MM_S` / `CART_MAX_ROT_DEG_S` are derived in the "speed
budget" section below rather than set here — they must stay consistent
with the joint-side limits and the controller's `VEL_RATIO`; keeping them
in two places guarantees drift.

## Control loop

**CONTROL_HZ = 250.0**: IK only costs 0.05 ms, so this isn't where the
bottleneck lives.

**XR_STALE_S = 0.25**: freeze threshold once no new XR frame has arrived
for this long. Originally 0.1, but `traj_c.jsonl` measurement (45 s)
showed a 13.9 ms median frame interval with a p99 of 122 ms — **61
dropouts over 100 ms in 45 seconds** (~1.4/s, mostly WiFi). At 0.1 s this
threshold was tripped constantly by normal jitter. Relaxed to 0.25 s
(≈18 normal frames): a real disconnect is second-scale and still caught,
while false triggers essentially disappear.

Relaxing this is safe: during an XR dropout, retargeting still targets
the "last known controller pose"; Cartesian rate limiting smoothly
converges the robot to that position and stops — the failure mode is
"stop," not "run away." If switched to a dedicated AP or wired link, this
can be tightened again.

**ARM_STATE = 1**: position-follow mode (high stiffness, default, simple
and predictable). Mode 3 (torque) is for contact tasks -- `--control-mode
impedance` flips this at runtime; impedance type/K/D live in
`algos.solver_config.ImpedanceConfig` / `configs/solver/impedance.yaml`
now, not here, since they aren't hardware-validated production values yet
(see docs/algos.md#impedanceconfig).

## Speed budget

Every speed limit in the whole chain lives in this section, all derived
from the single `VEL_RATIO` knob.

Why centralization is mandatory: these numbers used to live in three
places, independently hand-tuned:

| layer | value | |
|---|---|---|
| controller `set_vel_acc(10,10)` | 180×10% = 18 °/s | ← tightest |
| safety gate `MAX_JOINT_STEP_DEG=0.6/frame` | ×250Hz = 150 °/s | ← 8.3× wider |
| IK layer `MAX_JOINT_STEP_DEG×3` | 450 °/s | |

**The upper (software) layer was 8.3× wider than the controller — the
bottleneck sat on the controller itself.**

Consequence (Session D measurement, 2026-08-13, real hardware): `q_cmd`
permanently ran ahead of `q_meas`, tracking error accumulated
monotonically, and every single one of arm A/B's 15/6 follow attempts
ended in a `tracking_error` 5° lockout (lockout count == clutch count,
100%). Arm A dispatched only 197/15000 frames (1.3%) in 60 seconds.
Working the numbers back: at that arm A configuration, +X direction is
0.454°/mm, so 18°/s only reaches 39.7 mm/s at the end-effector; at
scale 0.5, any manual motion faster than 80 mm/s necessarily accumulates
error — and a natural hand sweep is 300–500 mm/s. So the 100% trigger
rate was a mathematical certainty, not a fluke.

Correct approach: **the upper-layer command-stream limit must be
strictly tighter than what the lower layer can execute.** This keeps
every command executable, so tracking error doesn't accumulate and
`tracking_error` protection goes back to catching what it's meant to
catch (collisions). This is why the fix (P0-2) was two things at once —
loosen the controller AND back-derive the upper-layer limits from the
controller's real capability. Doing only one makes things worse (either
hits limits faster, or gets slower) than the pre-fix state.

**JOINT_VMAX_DEG_S = 180.0**: from the PNVA table in
`ccs_m6_40.MvKDCfg` (this table is bit-identical in `ccs_m6_31`) — 180°/s
uniform across all seven axes; acceleration ranges 450–900 °/s².

**VEL_RATIO = 32**: controller-side joint follow velocity/accel percent
(dispatched via `arm_driver`'s `set_vel_acc`). To change gears, edit this
one number — the four derived quantities below are module-level
expressions recomputed on import. (`apply_vel_ratio()` is for runtime
changes, i.e. the `--vel-ratio` CLI flag; editing the file directly
doesn't need it.)

Ramp plan: 20 → 40, run each step for a full 60 s and confirm "0 lockouts"
in the stats before going higher. Stay clear of the arm's workspace.

⚠️ Raising velocity requires raising acceleration too — see `ACC_RATIO`:
ramp-up lag is v²/(2a), a **squared** relationship with velocity.
`ACC_RATIO` caps at 100, so under `MAX_TRACKING_ERR_DEG=5` the physical
ceiling on `VEL_RATIO` is ≈53% without also relaxing the tracking-error
threshold:
- vel 20 + acc 60 → lag 1.18°
- vel 40 + acc 100 → lag 2.82°
- vel 60 + acc 100 → lag 6.35° ❌ exceeds even at max acceleration

**ACC_RATIO = 100**: deliberately decoupled from `VEL_RATIO`, and set
much higher. Acceleration doesn't raise top speed (safety is governed by
`VEL_RATIO`) — it only determines how long the servo takes to catch up to
a velocity change. When the command stream steps to velocity v, the
position lag accumulated during the servo's ramp-up is v²/(2a):
- ACC 20% → J1/J2 only 90°/s² → lag 25.2²/180 = 3.53° (limit is only 5°)
- ACC 60% → J1/J2 270°/s² → lag 1.18°

Session D measurement (2026-08-13): with ACC=20, all four
`tracking_error` events were reported on joint 0 — i.e. J1/J2, which have
the lowest base acceleration (450°/s², other axes 900) — consistent with
this formula. Tracking error was caused by **acceleration**, not velocity.

**CMD_RATE_MARGIN = 0.70**: margin the command stream keeps below
controller capability. The 30% covers the servo's own follow lag, the
`ACC_RATIO` acceleration ramp (20% → 90–180 °/s²), and CAN bus jitter.

The four values below this point are **derived — to change `VEL_RATIO`,
go through `apply_vel_ratio()`**:

- `MAX_JOINT_RATE_DEG_S`: the **authoritative** ceiling; the safety gate
  converts it to a per-frame budget using `dt`.
- `MAX_JOINT_STEP_DEG`: nominal per-frame increment = rate × nominal
  period. Used only by the IK layer's branch-jump check and for logging;
  the safety gate itself uses `MAX_JOINT_RATE_DEG_S × actual dt`. This
  used to be a hardcoded 0.6°/frame — a per-frame quantity, not a
  per-second one — so a loop overrun silently changed the effective
  joint speed limit, and it was inconsistent with `CART_MAX_SPEED_MM_S`
  (already ×dt). Two dimensionally different quantities in the same
  chain is a bug incubator; unified to a rate.
- `MAX_STEP_DT_S`: cap on how much banked time a single step may cash in.
  A stalled loop (or an abnormal first-frame `dt`) must not convert
  banked time into one large jump — capped at 4 periods.
- `CART_MAX_SPEED_MM_S` / `NOMINAL_DEG_PER_MM = 0.2`: end-effector
  Cartesian speed limit. This is an **auxiliary** limit — authority
  remains `MAX_JOINT_RATE_DEG_S`, because the mm→° conversion varies
  sharply with configuration (measured on arm A: ±Z 0.133°/mm, ±X
  0.454°/mm, a 3.4× spread), so no fixed mm/s value can be correct in
  every direction. But if the Cartesian layer is much looser than the
  joint budget, every frame first gets rejected by IK as a
  `branch_jump` and then walked back by `solve_with_backoff`, wasting
  2–4 extra solves. `0.2°/mm` approximates a mid-range configuration so
  the two layers roughly align (at `VEL_RATIO=40` this reproduces the
  originally hand-tuned 250 mm/s).
- `CART_MAX_ROT_DEG_S`: end-effector rotation speed
  (`FOLLOW_ROTATION=True` only). Wrist-joint-to-end-effector-orientation
  is roughly 1:1, with a small margin. At `VEL_RATIO=40` this is
  ≈60°/s, matching the originally hand-tuned value.

## Safety thresholds

**LIMIT_MARGIN_DEG = 5.0**: minimum margin kept from the soft limit
(lowered from 8.0 for the 2026-09-10 runs). `robot.ini` sets
`VelLmtRange=8` — entering that range makes the controller itself start
rate-limiting. At 5.0 this layer lets the arm into the last 3° of that
range, so near a joint limit expect the controller's own limiting to show
up as follow lag.

**IK_POS_TOL_MM / IK_ROT_TOL_DEG**: FK back-substitution tolerance for an
IK solution — the last line of defense against a "fake success" solution.
Measured: without this check, `ik_nsp` returns a solution up to 116 mm
off target, with no error, once the target is outside the reachable
workspace, when the target is unreachable.

⚠️ Tolerance must be **much smaller than the per-frame Cartesian step**,
or a "doesn't move at all" solution gets judged a success: originally
0.5 mm, but at `vel 20%` each frame only travels
`CART_MAX_SPEED/CONTROL_HZ` = 0.504 mm. So when `solve_with_backoff`
backs off to `f=0.5`, the target is only 0.252 mm away, and a
zero-displacement solution with 0.252 mm error < 0.5 mm tolerance passes
validation (`res.ok=True`) while the arm doesn't move at all, every
frame reporting success. Reproduced offline (2026-08-13): 500 consecutive
"successful" frames with 0 mm displacement once at a workspace boundary.

`IK_POS_TOL_MM = 0.01` chosen because: measured FK back-substitution
error for genuine SDK solutions has median 7.2e-5 mm, max 1.05e-4 mm
(3000 samples) — 100× margin — while also staying more than 50× smaller
than the per-frame step at any speed setting.

**IK_MAX_STEP_DEG = 1.8**: the IK layer's per-frame joint-jump ceiling —
this is a **branch-jump detector**, not a velocity limit (velocity
limiting is the safety gate's job, via `MAX_JOINT_RATE_DEG_S × dt`; the
two have different responsibilities).

⚠️ This used to be `MAX_JOINT_STEP_DEG × 3`. When P0-2 shrank
`MAX_JOINT_STEP_DEG` from 0.6 to 0.1008, this threshold got dragged down
with it, from 1.8° to 0.302° — the detector got tightened along with the
speed budget, and started rejecting **legitimate large joint motion near
a singularity** as a branch jump too.

Why backing off doesn't help: near an elbow singularity, Δq ∝ √Δx, so
shrinking the target by 10× only reduces the joint requirement by 3.5×.
Measured offline (2026-08-13, arm A pushed to the J4=-7.1° boundary):
backing off just 0.0504 mm still required 1.265° of joint motion — all
four backoff levels exceeded 0.302°, so it rejected every one → couldn't
even back out of the boundary. Comparing 0.302° / 1.8° / 5.0° on the same
path: 0.302 pushes to 270 mm and gets stuck, unable to retreat; 1.8
doesn't get stuck within six seconds and can retreat to 120 mm; 5.0
would let a genuine branch jump through, sticking at 269 mm.

`1.8` chosen because measured genuine branch jumps hit 8.2° in a single
frame (median 2.16°, see `ik_solver` module notes) — 1.8 catches those,
while legitimate near-singularity motion is left to the safety gate,
which clamps it to a small step per `dt` (shows up as "slowing down").

**MAX_TRACKING_ERR_DEG = 5.0**: command-vs-measured deviation ceiling.
Exceeding it means the servo didn't keep up, or something was hit.

**MAX_TRACKING_ERR_S = 0.5**: how long the deviation must stay over
threshold before it's judged a real fault.

⚠️ Instantaneous overshoot and sustained overshoot used to be conflated.
Session D measurement (vel40/acc100, scale 0.45): all three lockouts sat
right at the line, 5.006–5.094°, while the live trace showed:
```
err 4.01 → 0.18 → 2.35 → 4.62 → [lockout] → 1.36 → 3.00 → 0.86
```
Error recovers on its own — that's tens of milliseconds of dynamic
jitter from hand direction changes (no jerk limiting anywhere in the
chain), not a servo that can't keep up. Responding to a transient by
disengaging and locking out makes the operator release and re-engage for
a single blip.

Current two-tier response:
- transient overshoot → don't dispatch this frame, `q_cmd` doesn't
  advance (backpressure); once the servo catches up, recovery is
  automatic and mostly invisible to the operator
- sustained overshoot → a real collision or servo fault; disengage and
  lock out

Backpressure is safer than lockout: not dispatching means `q_cmd` freezes
— the arm won't keep pushing into an obstacle — while a genuine collision
produces error that does **not** recover, which is exactly what this
timer catches.

**MAX_REJECT_S = 0.5** (→ `MAX_CONSEC_REJECT = MAX_REJECT_S × CONTROL_HZ`):
how long consecutive IK rejections are tolerated before disengaging.
Guards against "oscillating at a workspace edge."

⚠️ This used to be a frame count (15). At 250 Hz that's **60
milliseconds** — hitting a boundary for 60 ms triggered a lockout, while
human reaction time is 200 ms at best, nowhere near fast enough to
release and retry. Session D measurement (2026-08-13): arm A clutched 21
times and locked out 21 times in 60 seconds — each follow attempt
survived an average of 0.4 s. Same class of defect as
`MAX_JOINT_STEP_DEG`: a threshold expressed in "frame count" where the
loop frequency assumed at write time didn't match reality. 0.5 s = still
stuck against the boundary after half a second is what counts as
genuinely stuck.

**JOINT_LIMIT_OVERRIDE['B'][5]`**: arm B joint 6 limit is uncertain —
`tools/move_one_joint.py` reports ±58, the config file says ±60. Taking
the smaller value until Session C confirms.

## AXIS_MAP_UMI

The UMI rig's equivalent of `AXIS_MAP`, for recorded episodes replayed through
`scripts/umi_replay.py`. **Not an independent calibration** — it is `AXIS_MAP`
composed with the fixed Pico→world remap that
`xense-taccap-lerobot`'s `Pico4TrackerReader` has already applied to the
recorded poses (`PICO_TO_WORLD_R`, `[x,y,z] → [-z,-x,y]`):

```
v_base = (AXIS_MAP · Gᵀ) · v_world
```

Verified numerically: equals `AXIS_MAP @ G.T` to 0.0 for both arms,
determinant +1 for both. So it inherits `AXIS_MAP`'s confidence exactly — arm
A measured, arm B derived and still unverified. Do not "re-measure" this table
independently; if it looks wrong, `AXIS_MAP` is what's wrong.

Full derivation in `drivers.md#umi_sourcepy`.

## Gripper — UMI / TacCap follower

What is physically mounted on both arms now. Separate USB serial link per
side, nothing to do with Marvin's CAN pass-through, so none of the `GRIPPER_*`
values apply to it and none of these apply to the OmniGripper.

**Convention is inverted from `GRIPPER_*`: 0 = CLOSED, 1 = OPEN.** Deliberate
— that is what the SDK uses *and* what UMI episodes record, so a recorded jaw
value feeds `set_target()` with no conversion. The older path carries the flip.

**UMI_GRIPPER_MAX_TAU = 1.0 N·m** — abort-and-disable threshold, set by
operator decision on 2026-09-10 and not backed by a motor rating. The SDK
force-grasp example's 0.30 sits below the firmware's own 0.350 N·m auto-cal
stall torque and fired on normal grasps; at 1.0 it only catches a hard jam.
The OmniGripper's 8.0 is still no protection at all on this motor.

**UMI_GRIPPER_KP = 8.0 / KD = 0.3** — KP is the SDK's `ControlLoop` default.
The force-grasp example uses kp=5.0 for gentler contact.

**UMI_GRIPPER_GRIP_MARGIN = 0.036** — max lead of commanded over measured jaw
position, in *normalized* units. This is what bounds grip force:
`force ≈ KP × margin × stroke_rad` = 8.0 × 0.036 × 1.20 ≈ 0.346 N·m, level
with the firmware's 0.350 N·m stall torque. (The SDK example's 0.045 assumes
kp=5.0.) Same principle as `GRIPPER_MAX_OVERSHOOT_RAD` below, expressed
normalized because that is what the SDK and the recordings speak. The driver
prints the implied N·m at connect time rather than leaving it to be derived.

Note the vendor's warning that this gripper has a *position-dependent
restoring torque* (it springs toward open), so an absolute torque threshold
false-triggers as the jaws close on nothing. Contact detection should use
position stall, not torque — see `gripper_force_grasp_test.py`.

**UMI_GRIPPER_HZ = 100** — `ControlLoop` submit rate. Phasing
(`SubmitPhase.STREAM_LOCKED`, the default) matters more than the number; see
`drivers.md#umi_gripperpy`.

**UMI_GRIPPER_STALE_MS = 200** — ceiling on `observation().age_ms`. Motor
telemetry refreshes at ~50–100Hz, so past ~200ms the link is gone, not slow.

**UMI_GRIPPER_SN = {'A': None, 'B': None}** — pin the firmware SN per arm once
the grippers are labelled (`python3 drivers/umi_gripper.py --scan`). `None`
falls back to the SDK's side rule, which USB enumeration order can defeat.
A gripper driven as the wrong side is expensive to discover on hardware.

## Gripper — OmniGripper / DM4310 (deprecated)

No longer mounted on either arm; kept for the OmniGripper. Convention here is
0 = OPEN, 1 = CLOSED, the inverse of the UMI block above.


**GRIPPER_ENABLED = False**: off during Session A phase, enabled in P4.

**GRIPPER_HZ = 50.0**: its own thread, deliberately not folded into the
joint control loop.

**GRIPPER_OPEN_RAD / GRIPPER_CLOSE_RAD** — measured calibration
(2026-08-13, `gripper_range.py`, zero-torque free state, manually
worked back and forth three times):
- open end: -0.0444 / -0.0429 / -0.0444 rad → repeatability 0.0015 rad, a **hard** limit
- close end: +1.1183 / +1.1439 / +1.2270 rad → increases each trial, close side is **soft**
  (the jaws keep compressing after contact, no hard stop)
- total travel ≈ 1.30 rad

The old defaults (0.0 / 1.0) were copied from `tools/gripper_cycle.py`
and never actually calibrated. Direction was correct (q decreasing = jaw
opening) but neither endpoint was accurate.

Chosen values:
- `OPEN = -0.02`: open-end hard limit is at -0.044, leaving 0.024 margin.
  Deliberately not the measured minimum (-0.0582) — that was overshoot
  from the momentum of manual manipulation, not the steady-state limit;
  using it would command the gripper to sit against its hard stop
  continuously.
- `CLOSE = 1.15`: close end can be forced to 1.24, but only under applied
  force. Leaving 0.09 margin avoids fully-closed commands compressing the
  mechanism to its limit and producing enough torque to spuriously trip
  `GRIPPER_MAX_TAU`.

**GRIPPER_MAX_TAU = 3.0**: abort-and-disable threshold, carried over from
`tools/gripper_cycle.py`.

**GRIPPER_MAX_OVERSHOOT_RAD = 0.2** — force-limited position control: the
maximum amount the commanded position may lead the measured position, in
MIT mode grip force = `GRIPPER_KP × (commanded − measured)`, so this
value directly sets grip force:
```
grip torque ≈ GRIPPER_KP × GRIPPER_MAX_OVERSHOOT_RAD = 10 × 0.2 = 2.0 N·m
```

⚠️ Without this layer, gripping an object drives the command all the way
to `GRIPPER_CLOSE_RAD` while the measured position is blocked by the
object — deviation can exceed 0.6 rad → torque 6 N·m → exceeds
`GRIPPER_MAX_TAU` → gripper fault → `run_teleop.py` stops the whole arm.
I.e., "the first time it actually grips something, everything halts."

With this layer: once gripping, the command tracks the measured position
and grip force holds steady at 2 N·m, while genuine overload (gripping
something hard, mechanism jam) is still caught by `GRIPPER_MAX_TAU`.

## Tool

`configs/tool/<name>.yaml` (`drivers.tool_config.ToolConfig`) — one file
per mounted end-effector, fed to `Marvin_Robot.set_tool()` by
`scripts/set_state.py --state drag/release --tool <name>` so the
controller's torque-mode gravity feedforward accounts for the tool's
weight instead of treating it as an unmodeled load. See
`docs/scripts.md`'s gravity-compensation section for the full story
(this is the current best explanation for the "one joint sagged" symptom
logged under `set_state.py` above, though not yet hardware-confirmed).

Deliberately not CLI flags (`--tool-mass`/`--tool-com`) — a checked-in
yaml per tool is meant to be measured once and then maintained, not
re-typed from memory by whoever happens to be running drag mode that day.

Per-arm (`arms.A`/`arms.B` inside the yaml): the gripper's mount isn't
guaranteed symmetric between the two arms, so each gets its own entry.
`configs/tool/omnigripper.yaml` is the entry for `drivers/gripper.py`'s
OmniGripper (DM4310, 达妙):
- **`arms.B` (right) is filled in** (2026-08-16) — `m≈0.693kg`,
  `com_mm≈[-3.69,-1.45,64.15]` — from running the SDK's own
  `Marvin_Kine.identify_tool_dyn()` offline against pre-collected data
  that ships with the SDK for this exact tool
  (`CommonConfig/LoadData_ccs/LoadData/omini-gripper-tool-parameter/
  right-arm/`). Vendor reference-unit data for the same DM4310 model, not
  collected against this project's own physical robot — see the yaml's
  `source:` field for the full provenance note and a flagged discrepancy
  in the SDK's own output. `docs/scripts.md`'s gravity-compensation
  section has the full identification-workflow writeup (this is the
  current best explanation for the "one joint sagged" symptom logged
  under `set_state.py`, though not yet hardware-confirmed against real
  drag/release behavior).
- **`arms.A` (left) is filled in** (2026-08-17) — `m≈0.682kg`,
  `com_mm≈[-9.77,-15.03,71.84]` — from a live online identification pass
  on this project's own arm A (`scripts/tool_calib_collect.py` +
  `scripts/tool_calib_identify.py`, see docs/scripts.md), not vendor
  reference data like `arms.B` above. Noticeably different from `arms.B`'s
  numbers (mass ~1.5% lower, COM offset in a different direction) —
  consistent with the two arms' mounts not being mirrored, so the earlier
  "don't copy B's numbers to A" caution was warranted.

### `kine_offset` — the fingertip frame, not just a `set_tool()` field

`kine_offset` ([x,y,z,a,b,c] flange → tool frame) started as the
`kineParams` half of `Marvin_Robot.set_tool()`, which only affects the
*controller's* Cartesian targeting and so did nothing for a stack that
commands joints. It has a second reader now: `algos.tool_frame.ToolFrame`
takes it as the **flange → fingertip-midpoint** transform, so UMI replay and
`scripts/umi_filter.py` retarget at the point that touches the object. A
recorded UMI pose *is* a fingertip midpoint, so following it at the flange
swings the real fingertips through an arc nobody demonstrated — invisible
under pure translation, wrong as soon as the wrist turns. Full derivation and
the measured cost: `docs/algos.md#tool_framepy`.

**All-zero means NOT MEASURED, not "no offset".** `ToolFrame.load()` returns
`None` for a zero entry rather than treating a zero-length tool as a
measurement, and every UMI entry point prints a warning naming which
trajectory it is really judging. `configs/tool/umi.yaml` is in exactly that
state for both arms today — measure it with `scripts/tip_calib.py`.

`kine_offset_source` is required as soon as `kine_offset` is nonzero
(`ToolArmConfig`'s validator), same rule and same reason as `source`: a tool
frame nobody can trace is a tool frame nobody can check.

An arm with no entry fails validation loudly (`load_tool_config` lists
which arms *are* covered) rather than silently defaulting to 0 or
borrowing the other arm's numbers — an unregistered tool just means no
gravity comp for that arm (today's status quo); a wrong number feeds
incorrect gravity torque into a live, energized arm, which is worse.

## UMI_START_JOINTS

The UMI reference stack splits this job in two: `bimanual_umi_env.py` holds a
fixed `j_init` (`[0,-90,-90,-90,90,0]` for the UR5e) whose only purpose is to
start well-conditioned, and `eval_real.py` then has the operator jog to a
task-appropriate start with a SpaceMouse before handing over (C to engage, S
to stop). We have no SpaceMouse, so this one constant has to do both jobs.

**Not derived from HOME, deliberately.** HOME is a *parking* pose — folded in,
TCP only 174.5mm off the centerline. Clearance is a property of the *demo*:
relative mapping reproduces the demonstrator's displacement, so the room
needed is just the recorded excursion mapped into the base frame.

| arm | inward (-Z) sweep | HOME TCP Z | closest approach from HOME |
|---|---|---|---|
| A | 152.8 mm | +174.5 | **+22.2 mm** |
| B | 95.6 mm | +174.5 | +79.3 mm |

22mm is the 2026-08-25 near-collision. Both arms sweep toward small Z (their
base frames are mirrored, so "inward" is -Z for both).

**HOME is not ill-conditioned** — w=304.4, sigma_min=0.733, limit margin 30 deg
(bounded by J2, not the wrist). It is *not* the wrist-aligned singularity
`j_init`'s `wrist2=+90` avoids; adding J6=+30 barely moves the numbers. An
earlier reading of the incident blamed singularity; that was wrong. HOME's only
problem is clearance.

**How these were found.** Grid search over the reachable space at 10-degree
resolution (330750 arm configurations x neat wrist combinations), ranked by
manipulability subject to a joint-limit margin >= 25 deg and a TCP box
(reaching forward, spread outward, same height band as HOME). Every joint is a
multiple of 10 by construction: these values get read off a terminal and typed
back in, and a number like 88.23 is a transcription error waiting to happen.

| | A HOME | **A start** | B HOME | **B start** |
|---|---|---|---|---|
| joints | 90/-90/-90/-90/0/0/0 | **70/-80/-110/-70/0/20/0** | -90/-90/90/-90/0/0/0 | **-70/-80/110/-70/0/20/0** |
| TCP | 427.0/305.0/174.5 | 486.7/293.9/**384.4** | 427.0/-305.0/174.5 | 486.7/-293.9/**384.4** |
| distance from HOME | — | **219 mm** | — | **219 mm** |
| w | 304.4 | **312.8** | 303.7 | **312.8** |
| sigma_min | 0.733 | **0.782** | 0.731 | **0.782** |
| limit margin | 30.0 deg | **40.0 deg** | 30.0 deg | **38.0 deg** |
| followed | 2484/2484 | 2484/2484 | 2484/2484 | 2484/2484 |
| closest approach | 22.2 mm | **232.1 mm** | 79.3 mm | **289.2 mm** |

Better than HOME on every metric, not only clearance.

**B is A mirrored** (negate J1/J3/J5/J7, keep J2/J4/J6). The rule is verified
against both `HOME_JOINTS` and `HOME_WAYPOINTS` and reproduces a Y-negated TCP
exactly -- but B was still re-checked through the full pipeline on its own
rather than assumed correct from the mirror.

**Which base axis is vertical** is settled by `robot.ini`, not inferred:
`GravityY=+9.81` for arm A and `GravityY=-9.81` for arm B, so +Y is down for A
and -Y is down for B. HOME's TCP sits 305mm below the base origin.

**Re-derive when the episode changes.** The clearance requirement belongs to
the demo. `scripts/umi_replay.py`'s pre-flight prints the closest approach for
whatever start pose the arms are actually in.
