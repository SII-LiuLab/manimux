# Tianji differential IK

UMI/Tianji now selects `policy.options.ik_backend: analytic | diff`. The default
remains `analytic`: its SDK solution, FK tolerances, 1.8-degree branch check,
joint-limit margins and J6/J7 check are unchanged. The optional `diff` backend
is implemented in `src/manimux/kinematics/tianji_diff.py`, using the existing
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
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --checkpoint /path/to/trusted/pass_ball.ckpt \
  --ik-backend diff \
  --diff-ik-config configs/umi_dp/tianji/ik/diff.yaml \
  --bind-runtime-config data/experiments/pass-ball-diff.yaml
```

Add `--runtime-template configs/umi_dp/tianji/infra/pass_ball/rtc.yaml` for RTC.
The model, checkpoint identity and RTC sampler stay the same. IK selection is
an embodiment setting. The binder writes these effective runtime options:

```yaml
policy:
  options:
    ik_backend: diff
    ik_validation_dt_s: 0.004
    diff_ik:
      # max_velocity_rad_s and dt_max_s are inserted from execution.motion_limits.arm.
      w_pos: 1.0
      w_rot: 1.0
      lam: 0.001
      check_j67: true
      mu_nullspace: 1000.0
      nullspace_activation_deg: 25.0
      limit_margin_deg: null
      j67_margin_deg: null
      max_lag_mm: 5.0
      max_lag_deg: null
      lag_policy: report
```

`max_velocity_rad_s` comes from `motion_limits.arm.max_velocity`; `dt_max_s`
comes from its `max_step_dt_s`. There is no second hardcoded speed constant.
The history strategy checks these values against the currently loaded profile
on every construction, rejecting missing/stale/conflicting values. Rebind after
changing the profile. Both shared motion limits and a finite dt cap are required.

The QP uses independent per-joint velocity constraints regardless of the shared
executor's `per_joint`/`isotropic` mode; this matches the reference differential
solver. Executor shaping and the command guard still apply afterwards.
They may further alter the predicted joint trajectory.

Null joint-margin settings inherit this arm's configured kinematic margin
(5 degrees by default); the diff solver cannot lower that margin. The right
arm's J6 override remains ±58 degrees before the margin. `check_j67: false` is
rejected by the UMI adapter. Positive `j67_margin_deg` can override the QP buffer.
Optional `nullspace_weights` must contain seven finite nonnegative weights;
`nullspace_activation_deg: null` selects the original global quadratic cost.
Weights, regularization, activation, lag and margin settings are strictly validated.

## Algorithm and failure behavior

Each step solves the original velocity QP:

```text
minimize  ||J qdot - desired_twist||²_W + lam ||qdot||²
          + mu * dt * gradient(joint_limit_cost) · qdot
subject to joint-velocity and margin-adjusted position boxes,
           the linearized J6/J7 interference inequality.
```

External poses/joints use metres/radians. Internally the QP retains mm/degrees
to preserve the reference weighting and regularization. Its spatial orientation
error is a rotation vector, not an Euler-coordinate delta. The flange Jacobian
uses modified DH's post-link joint axes and includes the static flange offset.
Targets are transformed from TCP to flange using the same mounted tool as FK.

The solver caps each effective dt and clips numerical OSQP overshoot to the
velocity box. It rejects empty boxes before `OSQP.update` so OSQP cannot silently
reuse a stale problem. Unsolved statuses and nonfinite solutions fail; solved
steps are checked again for position margin and J6/J7 interference. Fixed sparse
patterns and within-chunk primal warm starts preserve the upstream QP procedure.

Differential IK is a rate controller: an unreachable target can produce a valid
small joint step with a large residual. The mandatory positive `max_lag_mm`
therefore defaults to CalibWrist's 5 mm guard. This residual is measured at the
**flange**, after the step; it is not a live TCP tracking monitor. Optional
`max_lag_deg` uses the reference maximum wrapped XYZ Euler-coordinate residual,
not a geodesic rotation angle. It defaults to null, as in CalibWrist.
`lag_policy` decides what a residual over these thresholds means. `abort` (the
library default) turns it into a `tracking_lag` failure. `report`, the default of
CalibWrist's `real_run` and of `configs/umi_dp/tianji/ik/diff.yaml`, keeps the
bounded step and only counts it; the adapter records each arm's worst residual
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
at-most-`ik_validation_dt_s` steps (also respecting the QP dt cap). Each step gets
its actual subdivision duration. The first knot uses its remaining target time
with the existing minimum validation interval; later knots use checkpoint dt.
Only final joint knots are retained. The shared executor then interpolates those
joint knots and sends commands on control ticks. CalibWrist samples TCP at the
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
This aggregate chunk work blocks the control thread, including executor smoothing;
it exceeds a 250 Hz tick's 4 ms budget. These are different measurements from the
duration of one IK solve. The new backend is offline integrated, with no claim
that the current whole control chain meets 250 Hz deadlines. Process decoding
with measured history and RTC still requires a separate runtime design review.

Twenty-three differential/adapter tests plus twelve existing UMI tests passed,
including finite-difference Jacobians, velocity and dt caps, invalid inputs,
empty boxes, J6/J7 conflicts/post-checks, lag guards, reset, profile conflicts,
real H16/H64 chunk decoding and atomic rejection on the final right-arm action.
The final combined viewer/session/config/executor/Tianji/camera/UMI/diff-IK and
mock-runtime regression suite passed 218 tests in 13.02 seconds. Ruff passed
on the changed source, scripts and tests. A real H16 checkpoint was bound with
`--ik-backend diff --diff-ik-config configs/umi_dp/tianji/ik/diff.yaml` in the
model environment; the runtime environment loaded the paired config, validated
the shared profile, and constructed both QP solvers with no torch or SDK loaded.
No hardware motion, real closed-loop policy rollout or task success was tested.
