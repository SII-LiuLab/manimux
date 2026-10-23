# Tianji differential IK

UMI/Tianji now selects `policy.adapter.ik_backend: analytic | diff`. The default
remains `analytic`: its SDK solution, FK tolerances, 1.8-degree branch check,
joint-limit margins and J6/J7 check are unchanged. The optional `diff` backend
is implemented in `manimux/kinematics/tianji_diff.py`, using the existing
`TianjiKinematics` DH chain, flange FK, tool transform and joint limits.

The port follows `SII-LiuLab/tianji-control` revision
`1e7dfdbc94c62f87501d6485b8c0e43ce6dbf513`, specifically `algos/diff_ik.py`,
`algos/kinematics.py` and `JointLimitAvoidance` in `algos/nullspace.py`.
The position-lag guard follows CalibWrist `deploy/tianji/ik_guard.py` and
`real_run.py` at `f17a62a2ab77207de54fdca73dc8ff4d76f87cdb`.
Production code imports neither checkout. OSQP, NumPy and SciPy are sufficient;
this backend loads no torch, robot connection or vendor IK library.

## Configuration

Install the runtime extra (the model environment does not need it):

```bash
uv pip install --python .venv/bin/python -e '.[xpolicylab,tianji-diff-ik]'
```

Use the same checkpoint binder as the analytic path:

```bash
envs/umi_dp/.venv/bin/python manimux/servers/umi_dp.py \
  --experiment manimux/configs/experiments/pass_ball/umi_dp/tianji_taccap_umi_dp_diff.yaml \
  --checkpoint /path/to/trusted/pass_ball.ckpt \
  --bind-runtime-config data/experiments/pass-ball-diff.yaml
```

The checked-in differential-IK experiment references the shared embodiment profile:

```yaml
policy:
  adapter:
    ik_backend: diff
    diff_ik:
      config: ../../embodiment/arm/tianji_diff_ik.yaml
```

Use the same reference from another Tianji experiment. The model, checkpoint identity and
RTC sampler stay the same. IK selection is an embodiment setting. The config loader expands
the profile, and the checkpoint binder writes these tuning options:

```yaml
policy:
  adapter:
    ik_backend: diff
    ik_validation_dt_s: 0.004
    execute_diff_ik_substeps: true
    diff_ik:
      w_pos: 1.0
      w_rot: 1.0
      lam: 0.001
      limit_margin_deg: null
      j67_margin_deg: null
      mu_nullspace: 1000.0
      nullspace_activation_deg: 25.0
      nullspace_weights: null
      max_lag_mm: 5.0
      max_lag_deg: null
      lag_policy: report
```

At runtime, chunk adapters and step IK use `resolve_diff_ik_config` with
`executor.motion_limits`. By default the QP inherits `motion_limits.arm.max_velocity`.
An explicit `diff_ik.max_velocity_rad_s` selects a tighter solver cap; it must be
finite, positive and no greater than the executor arm limit. Chunk tuning lives
under `policy.adapter.diff_ik`; step tuning lives under `execution.ik.diff_ik`.
Omitting the field preserves existing shared-rate behavior. The checkpoint binder
and history strategy do not copy it.
A finite executor `max_step_dt_s` is still required by the Tianji control profile,
but it remains an executor command-budget setting rather than a solver field.

The QP uses independent per-joint velocity constraints regardless of the shared
executor's `per_joint`/`isotropic` mode; this matches the reference differential
solver. Executor shaping and the command guard still apply afterwards.
They may further alter the predicted joint trajectory.
As of 2026-10-20, the pack-plate `checkpoint_comparison_20261019` catalog explicitly
uses pi/2 rad/s for both QP paths, pi rad/s for the executor and runtime command
velocity guard, and no command acceleration bound. Native joint policies bypass IK.
Installing this independent-cap capability requires restarting the serving runtime
once; subsequent changes to these YAML values can use idle Reload config.

Null joint-margin settings inherit this arm's configured kinematic margin
(5 degrees by default); the diff solver cannot lower that margin. The right
arm's J6 override remains ±58 degrees before the margin. The J6/J7 interference
constraint is mandatory and is not configurable. Positive `j67_margin_deg` can
override the QP buffer.
The pack-plate `checkpoint_comparison_20261019` study explicitly overrides each
arm's command/IK margin to 3 degrees as of 2026-10-20; its controller and executor
command boxes align with that margin. Null QP wrist margins also inherit 3 degrees.
Measured joint bounds are separately inset 0.5 degrees from the base range in both
the driver and runtime; they do not change IK constraints. Shared component defaults
remain at 5 degrees.
Optional `nullspace_weights` must contain seven finite nonnegative weights;
`nullspace_activation_deg: null` selects the original global quadratic cost.
Weights, regularization, activation, lag and margin settings are strictly validated.

## Acceleration-aware step execution (2026-10-23)

Step tuning additionally accepts `max_acceleration_rad_s2`; omission inherits the
executor arm acceleration cap. A finite executor acceleration cap cannot be paired
with null or a larger QP cap. This is a step-only field: chunk solvers do not have
executed command velocity history and reject it. With both acceleration caps null,
the existing step velocity-QP behavior is retained.

The step solver receives the last dispatched arm position and the direct executor's
nominal-tick command velocity explicitly. OSQP warm-start velocity is not this
history. In addition to velocity, the QP enforces per-joint acceleration and the
convex joint/wrist stopping envelope `q + T*v in P`, where
`T = max(dt, vmax/amax)` and `P` includes joint margins and all four buffered J6/J7
halfplanes. Because `q + dt*v + T*(1-dt/T)*v = q + T*v`, the backup velocity
`(1-dt/T)*v` preserves feasibility and changes velocity by at most `amax*dt`.
The nominal period is fixed; this is a command-space certificate, not a physical
tracking, jerk or wall-clock timing guarantee. It restricts speed near boundaries
without adding a fixed delay to the target stream.

The QP retains the original objective, lambda and nullspace settings. Numerical
reserves tighten position/wrist bounds by 0.001 degrees and rates by at most
0.001 degrees/s; OSQP uses `eps_abs=eps_rel=1e-7`, up to 4000 iterations.
Final acceptance checks the configured velocity, acceleration, current/next position,
**buffered** wrist region and stopping envelope with float64-roundoff tolerance only.
There is no independent joint clipping of an acceleration-aware QP solution.

The direct executor still applies its own motion limits. Configure these above the
QP limits so ordinary QP commands pass unchanged; the pipeline validates the final
executor output before dispatch. An unsafe modification raises and stops the runtime.
The effective executor velocity after `max_step_dt_s` must also cover the QP cap.
On either-arm failure, excessive lag under `abort`, or deadline rejection, both arms
use the certified backup and retain command velocity history. Timeline gaps use the
same backup, stopping outright once that fits the acceleration bound. Diagnostics use
`step_ik_brake`, `fallback: brake`, the original failure reason and solver details.
A state without a valid backup fails explicitly rather than clipping or inventing a
zero-velocity history. Lifecycle pause/Home/emergency handling remains separate and
reanchors only through the existing runtime reset path.

The 1022 pack-plate catalog's twelve Step variants select QP caps pi/2 rad/s and
80 rad/s^2, with outer executor caps pi rad/s and 160 rad/s^2. Other 1022 execution
variants retain their 80 rad/s^2 executor cap. Restart runtime for this implementation;
subsequent YAML edits may use idle Reload config. Offline checks:

```bash
.venv/bin/python -m pytest tests/unit/test_step_diff_constraints.py
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261022/launch.sh check
```

## Algorithm and failure behavior

### Optional step elbow branch protection (2026-10-23)

`execution.ik.diff_ik.elbow_guard_deg: 1.0` preserves the initial elbow side,
keeping J4 at least one degree away from full extension. It is disabled by
default, requires finite step acceleration limits, and is rejected by chunk
configuration. All twelve 1022 Step variants enable it. For the verified M6-40
DH geometry, full extension is J4 = -6.869643733 degrees, not zero. The arm's
`elbow_extension_rad` property rejects other DH geometries rather than applying
an unverified constant.

The first execution seed selects the side; subsequent model targets cannot change
it. The virtual J4 bound participates in the same QP stopping envelope, braking
backup and final-command validation as the physical limits. It does not alter
the physical-limit nullspace objective, lambda or global velocity/acceleration caps.
QP failure/reset and running timeline gaps preserve branch identity. Pause/Home
reset uses `reanchor=True`; the next execution anchor selects a new side. Warmup
also validates the anchor without dispatching commands. A starting J4 inside the
guard plus the 0.001-degree numerical reserve raises with a Home/reposition
instruction. There is no automatic projection, escape motion or branch switch.
Tasks requiring an intentional elbow-side transition need a separate transition;
ordinary pose tracking is not allowed to cross extension while the guard is active.

Frozen-target validation covers 28 1022 episodes (119,431 control ticks): eight
original acc=80 episodes and twenty earlier episodes standardized to acc=80.
The prototype removes all three previously observed large tracking regressions,
with no position/wrist violations or terminal failures. This is command-FK
tracking evidence, not measured hardware tracking or physical task success.
Tests cover both elbow sides, interior motion, startup rejection, QP reset,
lifecycle reanchor and braking. The one-degree guard reduces geometric maximum
wrist radius by about 0.023 mm; this is not a bound on arbitrary TCP tracking error.

Each step uses the original velocity objective with global wrist constraints:

```text
minimize  ||J qdot - desired_twist||²_W + lam ||qdot||²
          + mu * dt * gradient(joint_limit_cost) · qdot
subject to joint-velocity and margin-adjusted position boxes,
           all four measured J6/J7 interference halfplanes at q + dt*qdot.
```

External poses/joints use metres/radians. Internally the QP retains mm/degrees
to preserve the reference weighting and regularization. Its spatial orientation
error is a rotation vector, not an Euler-coordinate delta. The flange Jacobian
uses modified DH's post-link joint axes and includes the static flange offset.
Targets are transformed from TCP to flange using the same mounted tool as FK.

The solver uses the duration supplied by its caller and clips numerical OSQP
overshoot to the velocity box. It rejects empty boxes before `OSQP.update` so OSQP cannot silently
reuse a stale problem. Unsolved statuses and nonfinite solutions fail; solved
steps are checked again for position margin and J6/J7 interference. Fixed sparse
patterns and within-chunk primal warm starts retain the upstream update procedure.

As of 2026-10-20, the position-margin post-check allows only float64 roundoff:
16 machine epsilons times the largest absolute raw joint bound (about
6e-13 degrees for Tianji). This does not change the QP box or solver tolerances.
The wrist QP now includes all four halfplanes derived from `BD67_REAL`, rather
than only the starting quadrant. For the current table they are equivalent to
`1.025*abs(q6_next) + abs(q7_next) <= 110.5 - j67_margin_deg`.
This prevents a large step from escaping through an unrepresented quadrant.
Unsupported curved/nonconvex tables fail explicitly during construction.
OSQP still solves to its existing tolerances; clipping its velocity to the
independent joint box is not a projection onto the wrist polytope. The final
`j67_ok` guard checks the raw measured boundary, not the configured QP buffer.
Consequently, the buffer is not an exact post-solve guarantee.

Staged offline validation on 2026-10-20 used all 115 original pack-plate and
37 original bottle UMI episodes at native 30 Hz, with position/wrist margins
fixed at 6/5 degrees and no velocity or acceleration bounds. The frozen baseline
reproduced previous acceptance and float32 joint velocities for every episode.
Numerical-only rejection rates changed from 9.97% to 9.31% (plate) and
40.17% to 41.28% (bottle). Adding all wrist planes reduced rejection to zero
in both tasks, but speed P95 increased from 10.66 to 16.76 rad/s and from
40.62 to 61.52 rad/s respectively. Full-frame position-error P95 fell from
624.41 to 146.98 mm and from 911.63 to 573.18 mm. Precise pose coverage
(both arms accepted and within 5 mm / 5 degrees geodesic error) changed from
71.44% to 70.66% and 34.95% to 35.98%. No accepted raw wrist-boundary violations
occurred; the largest wrist-buffer shortfall was 0.301 degrees. These results
establish rejection/constraint improvements, not smoother motion or physical
task success. No damping, objective weights, margin settings or live services
were changed. Historical upstream parity results below predate these intentional
constraint and numerical-acceptance changes.

Differential IK is a rate controller: an unreachable target can produce a valid
small joint step with a large residual. The mandatory positive `max_lag_mm`
therefore defaults to CalibWrist's 5 mm guard. This residual is measured at the
**flange**, after the step; it is not a live TCP tracking monitor. Optional
`max_lag_deg` uses the reference maximum wrapped XYZ Euler-coordinate residual,
not a geodesic rotation angle. It defaults to null, as in CalibWrist.
`lag_policy` decides what a residual over these thresholds means. `abort` (the
library default) turns it into a `tracking_lag` failure. `report`, selected by
the pass-ball experiment and the reusable Tianji override, keeps the bounded
step and only counts it; the adapter records each arm's worst residual
and exceedance count in the chunk metadata as `diff_ik_lag`. Failed or empty QPs,
nonfinite solutions, joint margins and J6/J7 interference reject under both
policies, and any such failure rejects the entire predicted chunk.

Each arm resets both OSQP primal and dual state at the beginning of each
speculative chunk. The prior measured state seeds the new solve, so rejected
plans or a prior engagement cannot carry QP state into a new chunk. The original
CalibWrist reset only zeroed its previous velocity; this intentional difference
makes the new chunk independent of that hidden solver history.

## Timing and validation

IK runs while **decoding a predicted chunk**, with SE(3) segments divided into
at-most-4ms steps by default. The optional
`ik_validation_dt_s` adapter override changes that internal subdivision. Each source knot uses
the fixed policy action interval, and each QP substep gets its actual subdivision duration.
The model-facing chunk always retains its original action horizon and interval. With
`execute_diff_ik_substeps: false` (the default), only final joint knots drive the runtime,
preserving the previous behavior. When the option is true, the adapter also passes the
already-computed joint substeps to the runtime-only trajectory. The timeline trims and
samples that dense trajectory for execution while Viewer and RTC continue to consume the
original source knots. CalibWrist samples TCP at the
control rate, solves each sample seeded from the preceding command, and retains
all those dense joint commands; it can precompute an entire chunk, and its async
path has a separate sender thread. The difference is which samples are retained
and how sending is scheduled, not a requirement to solve IK live on every tick.
Residual checks on predicted substeps do not establish executed Cartesian path
equivalence.

Reproduce offline checks without any devices:

```bash
.venv/bin/python -m pytest tests/unit/test_tianji_diff_ik.py tests/unit/test_umi_dp_tianji.py
.venv/bin/python scripts/validation/tianji_diff_ik_parity.py \
  --reference /path/to/tianji-control --output /tmp/tianji-diff-parity.json
.venv/bin/python scripts/validation/umi_dp_tianji_decode_probe.py \
  --ik-backend diff --output /tmp/umi-dp-diff-decode.json
```

On the integration machine, OSQP 1.1.3, the real reference modules were compared
over 1,200 steps: both arms, five poses including all J6/J7 quadrants and active
nullspace bands, and dt values 0.5/4/8/30 ms. All verdicts agreed. Maximum joint
difference was `3.93e-5 degrees`; maximum position-residual difference was
`1.47e-6 mm`. Metre/mm roundoff can perturb OSQP's retained dual state, so this is
a numerical comparison, not bitwise equality; the probe uses a `1e-4 degree`
tolerance. Source SHA256 hashes and exact dependency version are in its report.

Single-step solver medians were 0.22–0.26 ms. Full two-arm **inline decoding**
took 65–66 ms for H16 and 260–262 ms for H64, on small reachable synthetic poses.
On 2026-09-14 the Python around the QP was reduced without changing the step:
closed-form rotation vector/Euler/Rodrigues instead of per-step scipy `Rotation`,
the rigid-transform check written out with np.allclose's own tolerances, and the
post-step flange Jacobian reused as the next step's start. A chained step dropped
from ~205 us to ~95 us (OSQP itself ~7–8 us); decoding took ~28 ms for H16 and
~112 ms for H64. Against the previous code, verdicts were unchanged and joint
differences stayed below 3e-13 rad over 7,684 recorded steps and 12 chunk decodes.
On 2026-09-21 the same treatment was applied to the composed layer, where the
assembled TCP kinematics had reintroduced per-substep work around the unchanged
solver: `kinematics.base.rigid_transform` now writes out the allclose/isclose
route it has always used (24 us to 3 us for identical verdicts over 7,909 cases,
including reflections and perturbations at each tolerance), `ik` validates the
target once instead of again inside `flange_target`, and the inverse mount/tool
offset is kept for one tool state. IK remains arm-only; the tool contributes the
constant TCP-to-flange change of frame, not a solve. A substep dropped from
~190 us to ~127 us and a two-arm H16 decode from ~55 ms to ~37 ms. Replaying six
frozen chunks (16 knots x 9 substeps, both arms) reproduced the previous joint
trajectories bit for bit, and the full unit suite's failure set was unchanged.
This aggregate chunk work must not block the control thread. The Tianji UMI
profile therefore decodes the complete chunk in two isolated per-arm processes,
then merges both results atomically. Both processes use the request observation
state as their IK seed.
The adapter converts every source knot with the fixed policy action interval;
the shared timeline alone removes rows that have expired at the actual commit
time, selecting the first source row at or after execution starts. Hardware-free
process/RTC tests verify that the control loop continues ticking during decode
and that either arm's failure rejects the whole chunk.

Before this main-layout adaptation, the refactor branch recorded twenty-three
differential/adapter tests plus twelve existing UMI tests passing,
including finite-difference Jacobians, velocity and dt caps, invalid inputs,
empty boxes, J6/J7 conflicts/post-checks, lag guards, reset, profile conflicts,
real H16/H64 chunk decoding and atomic rejection on the final right-arm action.
Its final combined robogui/session/config/executor/Tianji/camera/UMI/diff-IK and
mock-runtime regression suite passed 218 tests in 13.02 seconds. Ruff passed
on the changed source, scripts and tests. A real H16 checkpoint was bound with
an explicit differential-IK profile in the model environment; the runtime
environment loaded the paired config, validated
the shared profile, and constructed both QP solvers with no torch or SDK loaded.
No hardware motion, real closed-loop policy rollout or task success was tested.


## Execution-time step + diffIK (2026-10-16)

`execution.ik: {mode: step, backend: diff}` reuses `TianjiDifferentialIK` through
`TianjiStepDifferentialIK`. The policy adapter decodes poses inline; no chunk IK
or dense substeps run. At each control tick, convert the sampled TCP target to
flange using the composed geometry, solve once from the last dispatched arm command with the
nominal control period, and apply direct execution plus existing command safety.
The seed includes executor shaping; it is not the previous speculative IK result.
Runtime initializes/reanchors the command on startup and lifecycle transitions.
No separate joint-position integrator is retained in the solver, so failed ticks
resume from the held command. Analytic step IK still uses measured joint seeds.

Tuning lives under `execution.ik.diff_ik`; the joint velocity bound is resolved
from `executor.motion_limits.arm.max_velocity`. The QP's lag policy is explicit:
`report` accepts a bounded step with tracking lag, whereas `abort` rejects it.
The optional pose limiter is upstream, so reported lag is the solved command's
FK residual against that limited target, not measured robot tracking error.
Both arms share the IK budget; any failed or late update holds all arm
and gripper commands. QP warm state is discarded on rejected ticks, pause, Home,
timeline gaps and episode warmup. Successful ticks record `step_diff_ik`; rejected
ticks record `step_ik_hold`. Native branch-jump rejection is not used by this QP;
its rate, joint-margin and J6/J7 constraints remain active.
