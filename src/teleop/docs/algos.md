# algos/ — design rationale and empirical data

Pure computation layer: IK, null-space control, retargeting, safety
clamping. No hardware/SDK dependency except `ik_solver.py`, which wraps
the vendor kinematics library.

## ik_solver.py

### Why the wrapper layer exists

Three empirically-discovered issues, not theoretical concerns:

1. **With `zsp_type=1`, `ik_nsp()` must be called after `ik()`.** Otherwise
   `ik_nsp` fails immediately with a 2-axis singularity report
   (`m_Output_IsDeg[1]=True`). The vendor demo uses the `ik` → `ik_nsp`
   order, but the docs don't state it as a hard requirement.
2. **`ik_nsp()` returning `True` does not mean the target was reached.**
   Measured: once the target exceeds the reachable workspace, `ik()`
   already returns `False`, but `ik_nsp()` still returns `True` with a
   solution whose FK reprojects up to 116mm off target — the hand is
   here, the robot is 10cm away, with no error reported. Hence every
   frame must do an FK back-substitution check; this validation is the
   core of this module. Cost: 0.02ms, negligible.
3. **Solution branches can jump.** Adjacent-frame targets differing by
   only 1mm can still cause IK to jump to a different solution branch;
   measured single-frame jump of 8.2° (at 90Hz = 742°/s, while joint
   vmax is only 180°/s). Requires a continuity check against the
   reference configuration.

Additional: when the target is out of reach, `m_Output_RetJoint` still
holds residual values (FK reprojects 400+ mm off) — never skip
validation and use `RetJoint` directly under any condition.

Median per-frame total time 0.022ms, p95 0.024ms — IK is not the
frequency bottleneck; validation is affordable.

### `BD67_REAL` (J6/J7 self-interference)

Measured on real hardware 2026-08-12 via
`get_param('float', 'R.A{0,1}.CTRL.BD67xx{0,1,2}')`, reading the live
parameter table on both arm controllers. Arms A and B are identical:
`0, ±1.025, ±110.5` — corresponding to the `robot.ini` / `m6_40` set,
**not** the fitted coefficients parsed out of `ccs_m6_31.MvKDCfg` (that
set is `[0.018, -2.321, 108]` etc., origin unknown, never verified on
hardware). This mismatch is what eventually identified the machine as a
4.0 and got `config.KINE_CFG` moved off `ccs_m6_31.MvKDCfg` (2026-09-11);
the constant is kept explicit anyway, so it survives a wrong `KINE_CFG`.
`initial_kine()`'s `j67` parameter is used internally by the
native `fx_kine` library for limit checking (`FX_Robot_Init_Lmt`) — must
pass real hardware values, not the numbers from the m6_31 file. The four
rows are ordered `++, -+, --, +-`, matching the row-selection logic in
`j67_ok()`.

### `RETRYABLE` reasons

These reason codes all mean "target asked for too much"; backoff or
projection can rescue them:
- `JUMP`/`IK_FAIL`/`NSP_FAIL`/`FK_MISMATCH` — target too far or out of
  reachable space
- `JOINT_LIMIT`/`J67` — target pushes an axis into the limit margin or
  interference zone

Measured (`traj_c.jsonl`): J6 limit is ±60°, actual travel reached +52°,
falling within the 8° margin. Without backoff these 277 frames would all
be dropped; with backoff the arm visibly slows near the limit — the
operator's expected feel, rather than suddenly stopping.

### `limit_override` merge (in `ArmIK.__init__`)

After this merge, `self.lim_n`/`lim_p` is the single source of truth for
limits across the whole pipeline; both the safety gate and the
nullspace controller read these two directly.

Prior bug: IK used the raw PNVA values while the caller merged a
separate copy for the safety gate, causing a 2° blind zone on arm B's
J6 (IK allowed `|J6|≤52`, gate only allowed `≤50`): IK would produce a
solution, the gate would reject it as `joint_limit` — a latching-level
fault. Replay showed arm B lost 16% of frames (1569/9701) to this; arm A
never triggered it because it had no override — purely a two-arm
mismatch.

### `probing()` context manager

Purpose: the nullspace controller probes ±`probe_deg` arm angles each
frame (2 probes) plus the real solve — 3+ `solve_with_backoff` calls per
frame. Without masking, `stats`/`n_backoff`/`n_projected` get flooded by
probe calls: measured in one 60s run, `ok` counted up to 47769 while
there were only 11987 control frames — badly misleading for debugging.

### `solve()` branch-jump recovery (RetJoint fallback → AllJoint)

When RetJoint is rejected for `branch_jump`, the solver picks a more
continuous solution from all candidate solutions. CCS kinematics can
have up to 4 solution branches; the SDK docs say callers may pick freely
from `m_OutPut_AllJoint` as long as: not within ±0.05° of a 2-axis
singularity, all axes within limits, no J6/J7 interference — exactly
what `_validate()` already checks.

Measured significance: in `traj_b.jsonl`, all 1118 `branch_jump` events
were dominated by joint 4 (elbow), median magnitude 2.16°. Using only
RetJoint would drop these frames entirely; switching solution branch
usually lets tracking continue.

### `solve_with_backoff()`

Why needed: rejections cascade. The upstream 0.6°/frame clamp makes
`q_cmd` lag behind the IK solution, so next frame's target is even
farther from `q_cmd` → more likely to exceed limits → rejected again.
Measured on the same trajectory: without the clamp, IK success rate is
88.6%; with the clamp, it drops to 73.4% — the 15% difference is
entirely this cascade.

Backing off physically means "move less this frame" — equivalent to
temporarily lowering end-effector speed. Much better than freezing:
freezing feels like the robot is stuck, backing off just feels slower.

Near-singularity regions require several degrees of joint motion for
even 1mm of Cartesian displacement — motion is necessarily slow there;
this function expresses "can't go fast" as "goes slow" rather than
"doesn't go."

Isotropic backoff that still fails falls through to axis projection
(dropping one base-frame axis's displacement component) — see
`_solve_axis_proj`.

### `_solve_axis_proj()`

Why needed: backoff is isotropic — it shrinks the whole displacement
vector together, so if any single direction is maxed out, the other
directions that could still move get cut to 1/10 too — manifesting as
"hits a boundary and the whole frame freezes," which then feeds the
`RejectCounter` until it latches.

Reproduced offline (2026-08-13, pushing from the home configuration
toward each direction with a locked orientation) — proved this is
overly conservative: at the stuck point, probing all six directions with
a normal frame's budget usually finds four to five directions still
solvable, only the one currently being pushed fails. E.g. pushing -Z
into the J2 limit: ±X/±Y/+Z all remain solvable.

After projection, the feel is "can't go down anymore, but can still
follow left" — the expected feel at a workspace boundary. Returns `None`
only when every component projection also fails (truly no way out), and
the caller keeps the original failure.

Only translation is projected, not orientation: constraints almost
always appear in translation; orientation is already locked constant
when `FOLLOW_ROTATION=False`.

## kinematics.py

### Why it exists

The differential-IK prototype (`diff_ik.py`) needs a Jacobian, and later
WBC will need Jacobians at arbitrary intermediate links, not just the
flange. `Marvin_Kine` exposes `joints2JacobMatrix()` (flange-only) but the
brief was explicitly to build an independent Jacobian rather than lean on
more SDK internals — this module is pure numpy, takes an already-loaded
DH table, and never calls into the SDK itself. Only its self-test does,
purely as a validation oracle (`fk()`, `joints2JacobMatrix()`) — none of
the SDK's solving is used anywhere in this module.

### DH convention — determined empirically, not assumed

`cfg['DH'][arm_type]` is an 8x4 table for a 7-joint arm — one row too
many, and the config file doesn't document what the extra row is or
which DH variant (standard vs. modified/Craig) it uses. Guessing wrong
here would silently produce a plausible-looking but incorrect Jacobian,
so this was settled by brute-force cross-validation against
`Marvin_Kine.fk()`: tried {standard DH, modified DH} x {row 0 static,
row 7 static} x {static row applied first, static row applied last} — 8
combinations — over 20 random configurations, and took whichever
combination's max position error was numerically zero.

Result: **modified DH**, rows 0-6 are joints 1-7 (their `theta_offset` is
added to the joint angle), **row 7 is a static flange transform applied
last** (not a base offset, which was the a-priori guess). Confirmed to
match `Marvin_Kine.fk()`'s full 4x4 pose (position *and* orientation) to
8e-9 — floating point noise, not an approximation. Verified on both
`arm_type=0` and `arm_type=1` (the two arms' DH tables are independent
data, not mirrored by the code, so both were checked rather than assumed
symmetric).

### The off-by-one that a Jacobian will not tell you about by inspection

First implementation used frame `i` (the state *before* joint `i`'s own
transform) as the rotation axis for joint `i`'s Jacobian column. This is
wrong for modified DH: `Rz(theta_i)` is applied *before* `Tz(d_i)`, so the
frame whose z-axis equals the physical rotation axis is frame `i+1`, not
frame `i`. Using the wrong frame doesn't crash or look obviously broken —
it produces a Jacobian with joints 1 and 2's columns identical (since DH
row 0 has `alpha=0`, frame 1's z-axis coincides with frame 0's, and both
lie on a line through the differing origins that happens to cancel in the
cross product at that specific config). It would have been easy to ship
this and only notice much later as "the QP seems to ignore joint 2." The
bug was only caught by cross-validating against a finite-difference of
the module's own FK — this is why the self-test checks the Jacobian two
independent ways rather than trusting the closed-form derivation alone.

### Unit convention

Public API takes/returns degrees (matching the rest of the codebase —
`config.py`, `ik_solver.py`, `safety.py` are all degree-based), but the
DH math and cross-product physics are done in radians internally, then
converted at the boundary:

- Linear rows (0-2): mm per **degree** of joint (`analytic mm/rad
  result x (pi/180)`).
- Angular rows (3-5): degrees-of-end-effector-rotation-rate per
  **degree**-of-joint-rate — numerically *unchanged* from the mm/rad
  derivation, because both numerator and denominator are angle-rate
  ratios, and the deg/rad conversion factor cancels between them.
  Treating this as a *finite* xyzabc Euler delta (rather than a rotation
  *rate*) over one control period is a small-angle approximation; at this
  control loop's per-frame rotation budget (well under 1 degree — see
  `docs/config.md`) the resulting error is well under 0.01 degree, but it
  is an approximation and should be re-examined if used somewhere with
  larger per-step rotations.

Cross-checked against `Marvin_Kine.joints2JacobMatrix()`, which reports
linear part in **m/rad** and angular part in unitless **rad/rad**: the
measured ratio was exactly `0.001` on every linear element (mm-vs-m) and
exactly `1.0` on every angular element (confirming the angular part truly
is convention-independent, dimensionless direction cosines) — a clean,
fully-explained relationship, not a suspicious near-match that needed
explaining away.

### Validation summary (self-test, `python3 algos/kinematics.py`)

Three independent checks, all against real DH tables loaded via
`Marvin_Kine.load_config`, over 30-50 random in-limit configurations:

1. `fk()` vs. `Marvin_Kine.fk()` — full 4x4 pose, max error ~1e-7mm.
2. `jacobian()`'s linear part vs. finite-difference of this module's own
   `fk()` — max error ~1e-9 mm/deg (this is what caught the off-by-one
   above).
3. `jacobian()` vs. `Marvin_Kine.joints2JacobMatrix()`, after the unit
   conversion above — max error ~1e-8.

## diff_ik.py

### Why it exists

Prototype differential-IK controller: instead of one-shot solving for a
joint configuration that reaches a target pose (`ik_solver.ArmIK`), it
solves a per-frame QP for the joint *velocity* that best tracks the
target, with joint limits and velocity limits expressed as linear
constraints rather than enforced by post-hoc rejection. Not wired into
`run_teleop.py` — compared offline against `ArmIK` by `bench/compare_ik.py`.

```
min  ||J qdot - v_des||^2_W + lambda ||qdot||^2
s.t. qdot in a box derived from joint position limits (Taylor-expanded
     around q_prev) intersected with velocity limits
```

`J` comes from `kinematics.ArmKinematics`, not `Marvin_Kine.ik()`/`ik_nsp()`.
`Marvin_Kine` is still used for `mat4x4_to_xyzabc()`/`xyzabc_to_mat4x4()` —
coordinate-format conversion, not solving — exactly as `ArmIK` itself uses
it in `_validate()`.

### Bug: orientation error must be an axis-angle rotation vector, not an xyzabc coordinate difference

The first working version computed the orientation part of `v_des` the
same way `ik_solver.ArmIK._validate()` computes `rot_err` for its
tolerance check: per-axis wrapped difference of xyzabc Euler angles,
`[wrap(target_a - cur_a), wrap(target_b - cur_b), wrap(target_c - cur_c)]`.
That's a reasonable proxy *for a validation threshold* (are these two
orientations numerically close), which is all `ArmIK` uses it for. It is
**not** a valid stand-in for an angular-velocity vector, which is what
`jacobian()`'s rows 3-5 actually expect — a raw Euler-angle coordinate
difference only coincides with the true axis-angle rotation vector near
the identity rotation. This arm's working orientations are nowhere near
identity (e.g. `[-166, 56, -131]` deg at the test configuration), so the
mismatch is not a rounding error, it's a wrong direction.

Symptom: tracking a target with a *held-constant* orientation (zero
commanded rotation) still drifted — `rot_err_deg` grew from 0.0001 deg to
0.24 deg over just 10 frames at 250Hz, and kept accelerating (per-frame
achieved angular velocity climbed from ~0.02 deg/s to ~17 deg/s over the
same 10 frames). This is a closed-loop instability, not a bounded
approximation error: each frame's small residual gets fed back through
the `err / dt` proportional gain (effectively gain 250 at 250Hz) and
compounds, because the direction of the "correction" computed from the
wrong error representation isn't actually the direction that reduces the
true orientation error.

Fix: compute the orientation error as a proper rotation vector between
the target and current *rotation matrices*: `R_err = R_target @
R_current.T`, then `scipy.spatial.transform.Rotation.from_matrix(R_err)
.as_rotvec(degrees=True)`. This is what `jacobian()`'s angular rows
actually correspond to. After the fix: max rotation error over the same
60-frame test dropped from >0.24 deg (still climbing) to 0.00015 deg —
matching the position-tracking precision, not merely reducing the drift.
Verified on a second, independent scenario too: 2000 frames of combined
position + multi-axis-orientation sinusoidal tracking, 2000/2000 passed
validation, no drift.

The `_validate()` tolerance check still uses the same per-axis xyzabc
wrapped-difference convention as `ArmIK` (not the axis-angle vector) —
that's intentional, for apples-to-apples comparability with the analytic
solver's own rejection criterion in `bench/compare_ik.py`'s report. The
distinction is: axis-angle for anything that becomes a *velocity command
fed through the Jacobian*, xyzabc coordinate difference is fine for a
*post-hoc closeness check*.

### OSQP mechanics

**Fixed sparsity pattern for `update(Px=...)`.** `P = 2(J^T W J + lambda
I)` changes every frame (J changes with q), but OSQP's `update()` only
accepts new *values* for an already-declared sparse pattern — it can't
change which entries are structurally nonzero without a full re-`setup()`
(which re-factorizes and defeats the point of warm-starting). Converting
a dense `P` to CSC via the obvious `scipy.sparse.csc_matrix(dense_array)`
silently drops exact-zero entries, so the pattern can vary frame to frame
whenever some entry of `J^T W J` happens to land on exactly 0.0.
`_dense_symmetric_to_full_upper_csc()` always stores the *entire* upper
triangle (28 entries for a 7x7), zeros included, so the pattern is fixed
by construction and `update(Px=...)` is always valid.

**Silent failure on `update()` with `l > u`.** Discovered empirically,
not documented anywhere: if a bad box (`l[i] > u[i]` for some `i`) is
pushed via `prob.update(l=..., u=...)`, OSQP prints a C-level error to
stderr (`ERROR in osqp_update_data_vec: Problem data validation.`) but
**raises no Python exception and does not apply the update** — the
problem silently keeps its *previous* frame's bounds, and the next
`solve()` returns a normal `'solved'` status using those stale bounds.
For safety-critical control code this is exactly the failure mode to
design against (see `safety.py`'s "reject is safer than allow"
principle): `solve()` computes `lo_bound`/`hi_bound` and explicitly checks
`lo_bound[i] > hi_bound[i]` *before* ever calling `update()`, returning
`R_QP_INFEASIBLE` itself rather than trusting OSQP to catch it. This can
only happen if `limit_margin_deg` leaves an empty range at some joint
(the position-limit half of the box shrinks to nothing) — a degenerate
configuration issue, not a normal-operation case, but one the code no
longer trusts the solver to report on its own.

Status codes checked: `1` (`OSQP_SOLVED`) and `2`
(`OSQP_SOLVED_INACCURATE`) both count as success; anything else
(`3`=primal infeasible, `5`=dual infeasible, `7`=max iterations, `8`=time
limit) is treated as a solve failure. In practice a box-constrained QP
with `l <= u` is always feasible (an empty box is caught before ever
reaching OSQP), so a genuine non-solved status here should mean numerical
difficulty, not infeasibility.

**Timing.** Warm-started `update()+solve()` on this problem size (7
variables, 7 box constraints) measured at ~0.24ms median, ~0.33ms p99,
under 2.4ms worst-case observed over a 2000-frame combined
position+orientation tracking run — comfortably inside the 4ms/frame
budget at 250Hz, and above the 0.9ms TracIK anchor cited as a reference
point, though this includes Python/numpy overhead (Jacobian computation,
rotation-vector conversion) on top of the OSQP call itself, not just
solver time in isolation.

### Bug: validating a one-shot tolerance on an incremental controller

`_validate()` originally rejected a frame (`R_FK_MISMATCH`) whenever
`FK(q_new)` missed the *original* target by more than `pos_tol_mm`/
`rot_tol_deg` — copied directly from `ArmIK._validate()`. That's correct
for a one-shot solver (`ArmIK.solve()` is expected to reach the target in
a single call). It's the wrong criterion for an incremental controller:
`DiffIKSolver` only ever takes a rate-limited step toward a target, by
design, and when the target moves faster than the joint-velocity budget
allows — exactly what happens near a joint limit, where the achievable
Cartesian speed for a given joint-velocity budget degrades (see
`docs/config.md`'s speed-budget section) — not fully arriving within one
~4ms frame is normal, not a failure. It's the same situation `SafetyGate`
handles for the analytic pipeline via `clamped=True` (still `ok=True`),
not a rejection.

This wasn't just a theoretical concern: running `bench/compare_ik.py`
against `data/traj_c.jsonl` (the trajectory that stresses joint limits)
with the original criterion produced a 25% reject rate, **100% of it
`fk_mismatch`** — i.e., the QP was doing exactly what it should (safe,
rate-limited progress), and the validation was flagging normal tracking
lag as a solver failure. Fixed by dropping the target-distance check from
the pass/fail gate entirely; `pos_err_mm`/`rot_err_deg` are still computed
and returned on every result as tracking-lag diagnostics, they just don't
gate `ok` anymore. The only thing that still triggers `R_FK_MISMATCH` is
non-finite (`NaN`/`Inf`) output — a genuine solver malfunction, not slow
tracking. See `bench/compare_ik.py`'s findings below for what the reject
rate looks like once this is fixed (spoiler: it doesn't go to zero, and
the real remaining cause is more interesting).

### J6/J7 interference as a linearized hard constraint

`bench/compare_ik.py`'s first run against `data/traj_c.jsonl` found the
QP's box constraints (position/velocity limits) never included the J6/J7
self-interference boundary (`ik_solver.BD67_REAL`) — see that section
below for the original finding (21.3% reject rate, all `j67_interference`,
extended stretches frozen at 0mm/s). Fixed by adding one more linear
inequality to the QP, coupling columns 5 and 6 (`_j67_linearized_row`),
Taylor-expanded around `q_prev` every frame exactly like the position-
limit box is — same technique, applied to a curved two-variable boundary
instead of independent per-joint ones:

```
g(q6, q7) <= -margin_deg   (g is the signed distance past the boundary;
                             quadrant-dependent sign, see _j67_ok)
=>  c6*qdot6 + c7*qdot7 <= bound   (linearized around q_prev, /dt_eff)
```

This is a genuine **cost-vs-constraint** design decision, not the only
option — the alternative is a soft penalty term in the cost (same
mechanism used for the null-space secondary objective below). Constraint
was chosen deliberately: J6/J7 interference is a real mechanical
collision limit, not a preference, and a penalty term only *discourages*
crossing it — how much depends on the relative weight against the task
term that frame, which varies with how hard the operator is pulling. A
hard constraint gives an exact, weight-independent guarantee (subject to
the same linearization-accuracy caveat every Taylor-expanded constraint
in this codebase already has), matching how position/velocity limits are
already handled, and matching this codebase's existing "reject is safer
than allow" principle (`safety.py`). Penalty terms are the right tool
when something else already guarantees the hard limit (that's exactly
the null-space case below); they're the wrong tool when nothing else does.

**New infeasibility mode this introduces.** With only independent per-
joint box rows, one row can never conflict with another (rows 0-6 don't
share variables), so the only infeasibility check needed was a per-row
`l>u` scalar comparison. A row that *couples* q6 and q7 can conflict with
the box even when every individual row is well-formed — box ∩ half-plane
can be empty even though box alone and half-plane alone are not.
Verified empirically (not assumed) that OSQP's `solve()` correctly
reports `OSQP_PRIMAL_INFEASIBLE` (status 3) for this case, distinct from
the silent `update()` failure mode documented above for a malformed
single-row `l>u` — no extra feasibility pre-check needed, the existing
non-`_SOLVED_STATUS` handling already covers it.

**Fixed sparsity pattern extends to `A` now, not just `P`.** `_build_A`
always emits 9 explicit entries (7 box diagonal + 2 interference-row
entries at columns 5/6), including when a coefficient is exactly 0 —
verified empirically that `scipy.sparse.csc_matrix` built from explicit
`(data, (row, col))` triples keeps a stored zero rather than dropping it,
same as already relied on for `P`. `update(Ax=..., Px=..., ...)` patches
all three in one call.

**Result on `data/traj_c.jsonl`:** reject rate 21.3% -> **0.0%**
(previously 100% `j67_interference`, now none) — better than the
analytic pipeline's 2.6% (`ik_failed`, a different and much rarer failure
mode). The J6-window plot's "frozen at exactly 0mm/s for extended
stretches" pattern is gone; diff-IK's speed trace stays smoothly nonzero
throughout the same window. The earlier isolated 13.5ms solve-time
outlier did not reproduce in this run (new max 1.2ms) — plausibly related
to whatever was happening near the old hard-reject boundary, but this is
one data point, not a confirmed explanation. Self-test coverage:
`algos/diff_ik.py`'s `__main__` verifies the linearized row's math
directly (an already-margin-violating state must reject standing still
and under-retreating, accept retreating fast enough) and end-to-end (a
constrained solve from a tight J6/J7 state keeps the true nonlinear
`_j67_ok` check satisfied, not just the linear proxy).

### Null-space secondary objective (joint-limit avoidance as a soft cost)

Reuses `nullspace.JointLimitAvoidance`'s cost function (the same one
`NullSpaceController` uses for the analytic pipeline) as a **linear**
term in the QP's cost, via `mu_nullspace`/`_joint_limit_grad`. This is
the cost-term counterpart to the J6/J7 hard constraint above — see that
section for why interference got a constraint and this got a cost:
joint limits already have their own hard box constraint elsewhere in
this same QP, so a soft preference here only shapes *which* of the
already-safe solutions gets picked, it never has to be the only thing
preventing a violation.

**Mechanism.** Linearizing `mu*H(q_prev + qdot*dt)` around `qdot=0` gives
a linear term `mu*dt*grad_H(q_prev)` added to the QP's `q` vector — no
new variable, no new constraint, `P` untouched. `grad_H` is computed by
central-differencing `JointLimitAvoidance.cost()` (14 cheap evaluations)
rather than hand-deriving and separately verifying a closed form —
deliberately reusing the one already-tested cost function instead of
maintaining two formulas that could drift apart, at negligible extra
cost given how cheap `cost()` is.

**Verified this doesn't just wave hands at "it resolves redundancy."**
Self-test constructs a state where a joint sits within the activation
deadband of its limit and the target requires zero task motion (target =
FK(q_prev)): with `mu_nullspace=0` (default) the joint doesn't move at
all over 200 frames — nothing pulls it, matching the theoretical
prediction that a flat cost direction stays wherever it started. With
`mu_nullspace=1000`, the same joint's margin measurably improves (+0.945
deg over 200 frames in the tested case) while `pos_err_mm`/`rot_err_deg`
stay effectively zero throughout — confirms the linear term only spends
the null-space direction (which doesn't affect the task by construction)
and doesn't leak into degrading tracking.

**Configuration-dependent, same caveat `nullspace.py`'s own docs already
state** ("which axes the arm angle has leverage over is configuration-
dependent — it is not a general cure"): the self-test above only works
because a specific configuration (`HOME_JOINTS['A']` with J2 pushed
toward its limit) was picked *after* checking numerically (SVD of the
Jacobian) that the null direction has a large J2 component there. At the
`q0` "desk task" pose used by this file's other tests, the same push on
J2 produces almost no effect — verified during development, not assumed
— because the null direction there has very little J2 component.

**Real-world result:** enabled (`mu_nullspace=1000`) in
`bench/compare_ik.py`'s diff-IK pipeline for all three trajectories.
`data/traj_c.jsonl`'s "accepted-but-within-2x-margin" rate dropped
slightly, 45.3% -> 43.8%; the J6-window plot is visually almost
unchanged from the pre-nullspace run. This is the expected outcome, not
a disappointing one: `docs/config.md`'s nullspace section already
documents that **J6 itself has ~0 null-space leverage** at the home pose
("J1/J4/J6 not recoverable... a genuine DOF shortage, not a tuning
problem") — and `traj_c` specifically stresses J6. The self-test's
before/after comparison is the real evidence this mechanism works;
`traj_c`'s near-unchanged numbers are consistent with, not contradicting,
what was already known about which joints this technique can and can't
help. A trajectory that stresses J2/J3/J5/J7 instead would be the fairer
test of this specific addition's real-world value.

### Open tuning knobs — not solved design decisions

- **`W` (task weight, mm vs. deg in one cost).** Default `w_pos=w_rot=1.0`.
  Positions and orientations are different physical units with no
  canonical shared scale; the linear vs. angular Jacobian block magnitudes
  differ by roughly 5-10x at typical configurations (singular values ~12
  vs. ~1.5 at the test configuration). The rotation-vector fix above
  resolved the *instability*; it did not resolve the *weighting*
  question — `bench/compare_ik.py` should report how well tracking holds
  under a couple of different `W` choices, not assume the default is right.
- **`lambda` (minimum-norm regularization).** Default `1e-3`. Trades exact
  task-space tracking for numerical conditioning through redundancy/
  near-singular configurations — the standard damped-least-squares
  tradeoff. Even when the desired twist is exactly achievable (non-binding
  constraints, well-conditioned J), `lambda > 0` means the QP will not, in
  general, drive the task residual to exactly zero. This is expected
  behavior, not a bug, but it does mean the differential solver's
  steady-state tracking error is a tunable, not a fixed property the way
  `ArmIK`'s FK-validated one-shot solve is.
- **`mu_nullspace`** (default `0.0`, off). Now implemented — see the
  "Null-space secondary objective" section above. `lambda ||qdot||^2`
  alone is still a minimum-norm term, not limit-aware; `mu_nullspace`
  adds the limit-aware push on top, opt-in.

## bench/compare_ik.py

### Methodology

Runs `ik_solver.ArmIK` and `diff_ik.DiffIKSolver` as two **fully
independent** closed-loop pipelines (own `Retargeter`, own `SafetyGate`,
own evolving joint state) over the same raw resampled XR frame stream —
not the same per-frame target list. `Retargeter.update()` reads `q_now`
*every frame* (not just at clutch-engage) as the origin of its Cartesian
rate limiter, so retargeting is genuinely closed-loop against whichever
solver is driving it: if the two solvers' joint trajectories diverge,
their target streams legitimately diverge too. That's the realistic
question ("if I swapped solvers, how would the whole closed loop
behave"), not an experimental error to control away.

The analytic pipeline's `NullSpaceController` stays disabled: it needs
reentrant "probe" solves that `DiffIKSolver` doesn't support, and its
mechanism isn't equivalent to `DiffIKSolver`'s own redundancy handling
(`mu_nullspace`, a cost term added directly to the QP — see the
"Null-space secondary objective" section above), so enabling it on the
analytic side wouldn't make this an apples-to-apples redundancy
comparison. `mu_nullspace=1000` IS enabled on the diff-IK pipeline.

Both solvers' raw output is passed through the *same* `SafetyGate`
instance-per-pipeline (same thresholds), matching how `core/arm_channel.py`
actually wires things — the gate is solver-agnostic in the real
architecture, so it should be in the comparison too.

### Findings on `data/traj_c.jsonl` (7719 engaged frames — the trajectory that stresses J6 toward its limit)

**Update:** the interference gap found here has since been fixed —
`diff_ik.py`'s "J6/J7 interference as a linearized hard constraint"
section above has the current numbers (reject rate now 0.0%, down from
the 21.3% described below). Left as-written below because it's the
finding that motivated the fix and the reasoning still explains *why*
the gap existed; read it as history, not current behavior.

**Reject rate: 2.6% (analytic) vs. 21.3% (diff-IK), and they're not
comparable failures.** Analytic's rejects are `ik_failed` (genuinely no
IK solution found). Diff-IK's are **100% `j67_interference`** — the QP's
box constraints cover joint position and velocity limits (both linear,
built into the QP directly, as intended), but the J6/J7 self-interference
boundary (`ik_solver.BD67_REAL`, a curved region in (q6,q7)-space) isn't
a QP constraint at all. Nothing in the optimization steers away from it;
it's only caught post-hoc in `_validate()`. `ArmIK`, by contrast, gets
multiple chances to route around the same boundary via
`solve_with_backoff`'s isotropic backoff and axis-projection retries
(`R_J67` is in `RETRYABLE`). `DiffIKSolver` has no equivalent retry —
one QP solve per frame, no second attempt. This is a real, fixable gap
in the current formulation, not a fundamental limitation of the QP
approach: the natural fix is linearizing the interference boundary
(gradient of the `BD67_REAL` quadratic in `(q6, q7)`) into an additional
row of the box/inequality constraint, the same Taylor-expansion trick
already used for position limits.

**The J6-limit-approach window plot is the most concrete evidence.**
(`bench/results/j6_window_traj_c.png`) By the window centered on
analytic's tightest J6 margin, the two pipelines have already diverged
substantially — diff-IK's J6 margin sits around 30-55 deg for this whole
window, nowhere near analytic's ~10-15 deg. And diff-IK's Cartesian speed
trace shows **extended stretches at exactly 0 mm/s** — fully stuck, not
just slowed — precisely because a rejected frame has no retry and no
interference-avoidance, so `q_cmd` freezes until the *operator's own*
motion happens to move the target out of the interference region again.
Analytic's speed trace also drops during rejection stretches, but
recovers with a sharp lurch, backed by the retry logic finding *some*
usable solution more often. In this specific window, **the unconstrained
QP behaves worse for the operator than the hard-reject analytic
pipeline** — the opposite of the naive expectation that soft QP
constraints are strictly smoother than hard rejection. That expectation
holds for the constraints that ARE in the QP (position/velocity limits);
it does not hold for the one that isn't (interference).

**Solve time:** diff-IK p50/p99 (0.25/0.28ms) comfortably beats the
0.9ms TracIK anchor and stays inside the 4ms/250Hz budget, as expected
given warm-starting. One isolated outlier at 13.5ms (frame 2210, `ok`,
J6 margin 40 deg — nowhere near the interference region, so unrelated to
the finding above) was not correlated with any constraint proximity or
solver state checked so far; worth investigating before considering this
production-real-time-safe, but it's a single frame out of 7719, not a
systematic pattern.

**On `data/traj_b.jsonl`** (55 engaged frames, doesn't stress limits) and
`--synth` (2003 engaged frames, gentle sinusoid): both solvers post 0%
reject and near-identical step distributions — expected, this is the
"easy" regime where the two approaches shouldn't differ, and the numbers
confirm they don't.

### Reading the reports

`bench/compare_ik.py <traj>` writes `bench/results/report_<name>.md`
(the four comparison axes as tables), `compare_<name>.csv` (every
resampled frame, both solvers, one row each — for anything not covered
by the report), and two PNGs (`step_histogram_<name>.png`,
`j6_window_<name>.png`). Outputs are gitignored — regenerate with
`.venv/bin/python3 bench/compare_ik.py data/traj_c.jsonl`.

## solver_config.py

### Why a typed config layer, and why not Hydra

`run_teleop.py` needed a way to pass tunables (`w_pos`, `mu_nullspace`,
...) into `DiffIKSolver` without either hardcoding them or exploding
`argparse` into one flag per parameter — and this is only going to get
worse once impedance control and WBC land, each with their own tunable
surface. The considered alternative was Hydra: it wasn't picked because
its main value (multirun sweeps, launcher plugins, deep config
composition trees) mostly doesn't apply to a hand-written real-time
control script, and `@hydra.main` takes over the program entry point
(changes the working directory, owns config resolution) for benefits
this project wouldn't use — `run_teleop.py`'s `main()` already has
carefully-ordered custom gates (`CALIBRATED`, now the diff-IK dry-run
gate) that are simpler to keep hand-written.

What actually addresses "keep this maintainable as more algorithms show
up, and catch bad parameters early" is a **repeatable pattern**, not a
framework: one `pydantic.BaseModel` per algorithm (field constraints do
the actual parameter cleaning — type coercion, `ge=`/`gt=` range checks,
`extra='forbid'` to reject an unrecognized key instead of silently
ignoring a typo), one YAML file per algorithm holding tuned defaults, and
one generic loader (`load_solver_config`). Adding impedance control
later is "write `ImpedanceConfig`, write `configs/solver/impedance.yaml`,
add one line to `run_teleop.py`'s `_SOLVER_CONFIG_MODELS` dict" — no
other plumbing changes, and nothing about `DiffIKConfig` or its YAML
needs touching.

The **existing, hardware-validated** analytic solver deliberately does
NOT use this layer — its tunables already live in `config.py`, and
migrating proven production values into a new mechanism for consistency
alone would be pure churn with no benefit. This layer is only for
solvers still being validated.

### `DiffIKConfig`

Mirrors `DiffIKSolver.__init__`'s keyword arguments exactly (see
`diff_ik.py` above for what each one does). `limit_margin_deg`/
`j67_margin_deg` default to `None` in both the model and the YAML file
(commented out there) rather than duplicating `config.LIMIT_MARGIN_DEG`'s
value — the fallback is resolved by the caller (`ArmChannel`), keeping
this module free of a `config.py` import and keeping the margin's single
source of truth actually single.

### `ImpedanceConfig`

Mirrors `drivers.arm_driver.ArmDriver`'s torque-mode SDK calls: `type`
(1=joint, 2=cartesian — 3=force control is rejected, `ArmDriver` doesn't
implement its different call sequence), `joint_k`/`joint_d` (7 values,
J1-J7), `cart_k`/`cart_d` (7 values,
`[transX,transY,transZ,rotX,rotY,rotZ,nullspace]`), `rot_type`/
`cart_ctrl_para` (Cartesian-only, see below). `joint_k`/`joint_d`/
`cart_k`/`cart_d` are all always required regardless of `type` — mirrors
the MarvinPlatform GUI's "Impedance parameter settings" panel, which
saves both joint and Cartesian rows together — so switching `type` later
doesn't need a new YAML.

`type=1` and `type=2` share the same call sequence
(`set_impedance_type()` + `set_joint_kd_params()` + `set_cart_kd_params()`,
all in one `clear_set()`/`send_cmd()` batch) — both K/D pairs are always
sent regardless of which `type` is active, matching the GUI mirroring
described above. `type=2` gets exactly one more call in the same batch,
`set_EefCart_control_params()` (`OnSetEefRot_A`/`_B`) — see `rot_type`
below for why. (An earlier version of this fix tried routing `type=2`
through a *different* SDK class entirely,
`Concise_Marvin_Robot.set_imp_cart_state` — reverted before it ever ran:
that class opens its own independent `.so`/`Connect()`, not the
`Marvin_Robot` instance `RobotConnection` already holds, so calling it
from `ArmDriver` would either crash — confirmed locally,
`AttributeError: 'Marvin_Robot' object has no attribute
'set_imp_cart_state'` — or, had it been on the right object, risked a
second connection on the SDK's documented one-connection-per-process UDP
port. `set_EefCart_control_params` is on `Marvin_Robot` itself, the
class this codebase actually uses everywhere else.)

No upper bound on any K field (only `ge=0`): the vendor SDK doc states a
Cartesian translation-stiffness range of 0~1200, but the values seeded in
`configs/solver/impedance_joint.yaml`/`impedance_cartesian.yaml`
(`cart_k[0:3]=3000`) come from the MarvinPlatform GUI and are confirmed
by manual GUI operation to run correctly on real hardware (screenshot,
2026-08-14) — the doc's range is demonstrably wrong, not a validation
ceiling worth enforcing.

Two YAML files, not one: `impedance_joint.yaml` (`type=1`) and
`impedance_cartesian.yaml` (`type=2`) — same K/D values in both (same GUI
panel, saved together), only `type` differs. `run_teleop.py`'s
`--impedance-type {joint,cartesian}` picks which file
`--impedance-config` defaults to, and cross-checks it against the
loaded file's own `type` field (refuses to start on a mismatch, e.g. a
hand-edited file whose `type` no longer matches its filename) — this
follows the same "don't trust two things to silently agree with each
other" principle as `ik_solver.py`'s `limit_override` merge bug.

Unlike `DiffIKConfig`, impedance control has no offline validation path
equivalent to `bench/compare_ik.py`: its whole point is physical
compliance behavior (response to real contact/gravity/inertia), which
can't be exercised without hardware. So there's no "dry-run stats vs.
offline numbers" gate the way `diff_ik` had — and `--dry-run` doesn't
even reach `ArmDriver.prepare()`'s torque-mode branch at all (that
function returns early for `dry_run`), so a clean `--dry-run
--control-mode impedance` run only proves config loading/CLI plumbing
works, not that the SDK calls are correct. First live validation is via
`run_teleop.py` directly (single arm, short duration, clear workspace),
not a separate offline/dry-run step.

`type=1` (joint impedance) is what's wired in first, not `type=2`
(Cartesian, `config.py`'s old default) — joint impedance is easier to
isolate (each joint independent) for software's first time driving this
control path; Cartesian involves an internal Jacobian-based conversion
that's harder to reason about near singularities. `ArmDriver.prepare()`
originally only implemented the Cartesian call
(`set_cart_kd_params`) — the joint call (`set_joint_kd_params`) had to be
added; before that fix, joint impedance wasn't actually wireable at all
regardless of `type`. This prediction held: joint impedance validated on
hardware with no issues; Cartesian didn't, and the reason turned out to
be `rot_type` (next).

### `rot_type`/`cart_ctrl_para` — the actual Cartesian bug

`type=2`'s first live attempts had unexplained behavior on the "other"
(redundant/coupled) joints when force was applied at the end-effector.
Root cause, found by comparing our call sequence against
`DEMO_C++/showcase_eef_cart_impedance.cpp` — a demo purpose-built for
exactly this scenario (its own header comment: "该 DEMO 用于【左臂：设置
笛卡尔阻抗/力控相关参数 + 切换扭矩模式 + 读回校验】的示例"): it always
pairs `OnSetCartKD_A(K, D)` with `OnSetEefRot_A(type, Dir)` in the same
`clear_set`/`send_cmd` batch. Our original `type=2` path called only the
former (`set_cart_kd_params`, which has no rotation-reference parameter
at all) and never the latter — so the end-effector orientation reference
was left in whatever state the controller happened to already be in,
undocumented and unverified.

This matters specifically for Cartesian, not joint, impedance: the
Cartesian control law computes a 6D end-effector pose error (translation
+ rotation) and converts it to joint torques via `τ = Jᵀ(q)·F` plus a
separate null-space term for the 7th (redundant) DOF — if the rotation
half of that error is built on an undefined reference, the error
propagates through the Jacobian transpose into torques on joints that
look, from the operator's side, unrelated to what's being pushed. Joint
impedance has no equivalent step (`τ_i = K_i·(q_i,target − q_i,measured)`
per joint, no Jacobian, no orientation reference at all), which is
consistent with why only the Cartesian path showed the problem.

**Fix**: `ArmDriver.prepare()`'s `type==2` branch now also calls
`set_EefCart_control_params()` (`OnSetEefRot_A`/`_B`) right after
`set_cart_kd_params()`, in the same batch — matching the C++ demo's call
order exactly (see `docs/drivers.md`). `rot_type=2` ("system
auto-calculated") + `cart_ctrl_para=[0]*7` is the demo's own value, and
the only one used here — **not** the value in an earlier version of this
fix (`0`, "undefined"), which came from a *different* vendor function's
docstring (`Concise_Marvin_Robot.set_imp_cart_state`'s `rot_type`
parameter) that turned out not to apply to the call this codebase
actually makes.

**The two vendor docstrings that describe this parameter disagree with
each other, which is exactly why only a real runnable demo — not either
docstring — was trusted for the final value:**
`Concise_Marvin_Robot.set_imp_cart_state`'s `rot_type`: `0`=undefined,
`1`=user-defined direction, `2`=system auto-calculated.
`Marvin_Robot.set_EefCart_control_params`'s own `fcType` (what
`ArmDriver` actually calls): `1`=user-defined direction,
`2`=system auto-calculated, `3`=used together with
`set_force_control_params` — no documented `0` at all. The two schemes
happen to agree that `2` means "system auto-calculated," which is one
more reason `2` was chosen over guessing at `1`'s `cart_ctrl_para`
format (itself underspecified — "前三个参数置为末端基于基座 X Y Z 顺序
的旋转" doesn't say what units/convention). Don't change `rot_type` away
from `2` without new evidence, the same standard this codebase already
applies to every other unverified SDK doc claim (see `ik_solver.py`'s
`BD67_REAL` section for the precedent: measured on real hardware, not
taken from the doc/config file).

## nullspace.py

### Redundancy geometry

Measured on arm A (2026-08-13): at one fixed end-effector pose, sweeping
the arm angle from -20 to +20 deg swings J2 from -104.4 to -75.6 deg. J2
is exactly the joint that saturates when pushing -Z, so the redundancy
is worth spending on joint-limit avoidance rather than leaving pinned at
zero.

### `JointLimitAvoidance` — activation form vs. classic form

Measured on arm A from the home pose against a locked arm angle
(2026-08-13): the classic (unactivated) form gains 8% of total reach but
loses 10% on +Z, because it keeps spending the arm angle in directions
that are reach-limited rather than limit-limited. With
`activation_deg=25`, no direction regresses and -Z still gains 48%.

`weights` exists because the joints are not equally roomy — J6 is ±60°
here, half of everything else — so a degree lost on J6 costs more.

### `NullSpaceController` — probe-based gradient

The angle→configuration map lives inside the SDK's analytic IK, so the
descent direction is sampled by probing ±`probe_deg` instead of
differentiated. Costs two extra IK calls per frame (~0.05ms), negligible
against the 4ms control period.

`probe_deg` must stay small enough that the induced joint motion clears
the IK layer's branch-jump threshold. Measured sensitivity is 0.4–1.1
deg of joint per deg of arm angle, so a 2° probe can move up to 2.2° of
joint — past `config.IK_MAX_STEP_DEG` (1.8°) — causing the IK to fall
back to a solution picked without the arm angle. The probe then returns
the same configuration for every angle and the gradient vanishes
silently; `n_blind` exists to surface this.

## retarget.py

### Design: relative mapping + clutch, not absolute mapping

Absolute mapping (controller pose directly as robot target) has two
fatal problems:
1. At the instant follow engages, the robot would jump instantly from
   its current pose to wherever the controller is — an unconstrained
   jump, the single most dangerous moment.
2. The operator's arm reach is limited; unreachable poses become
   uncontrollable, with no way to "release and recenter."

Relative mapping + clutch solves both: on clutch-press, latch
`(controller pose T_xr0, robot pose T_rob0)`; thereafter
`target = T_rob0 ⊕ scale · (controller displacement/rotation relative to T_xr0)`.
Releasing the clutch stops following and holds position; pressing again
re-latches. This lets the operator "lift and reset" like a mouse, so
reach is no longer a hard limit.

### Coordinate frames — two-stage, independently verifiable

Stage 1: raw XR data → right-handed frame. (Code originally assumed a
Unity-style left-handed frame needing a flip; Session B measured that
PICO 4U's pose is already right-handed, so no flip is needed —
`XR_HANDEDNESS_FLIP_AXIS=None` skips `flip_handedness` entirely.)

Stage 2: right-handed XR frame → robot base frame (a pure rotation, a
signed permutation with det=+1).

Applying a det=-1 matrix to transform orientation is wrong — a
reflection is not a rotation and mirrors the pose. `build_axis_matrix()`
explicitly checks `det≈+1` so this error can't pass silently.

Calibration source: XR-side convention measured in Session B, robot
base-side convention measured in Session C; combined into
`config.AXIS_MAP`. Until calibrated, `config.CALIBRATED=False` and
`run_teleop` refuses to go live.

### `build_axis_matrix` key naming

Keys are deliberately `xr_x`/`xr_y`/`xr_z`, not semantic names like
`right`/`up`/`forward`. An earlier version used semantic names, but PICO
4U's +Z measured as "backward" not "forward" (Session B) — so a key like
`'xr_forward'` actually meant "backward," with the sign silently
flipped. This "name says one thing, value means another" mistake is
nearly impossible to catch in the field. Axis names are objective;
semantic names can lie — hence axis names only.

### `Retargeter.arm` — per-arm axis mapping

The two arms have different local frames (GRV measured the Y axis is
mirrored between them), so the axis mapping must be per-arm — see
`config.AXIS_MAP`.

### `_fault_latched`

Why needed: without this latch, `disengage()` would be immediately
undone next frame by a clutch button that's still held down — measured
in a 5-minute dry-run, a single `tracking_error` became 12207 repeated
triggers; on real hardware this is equivalent to "hit something and
keep pushing into it."

### `engage()`: `_cmd_p`/`_cmd_R` start from current robot pose

Rate-limited following starts from the robot's actual current pose — so
the instant the clutch engages, the command equals the current state
with zero jump (this is exactly the problem relative-mapping + clutch is
meant to solve).

### `update()`: Cartesian rate limiting

Without this layer, a fast hand motion gets rejected by IK as
`branch_jump`, and once rejected the commanded configuration freezes
while the hand keeps moving — the gap only grows. Measured: IK success
rate collapses to 5.88% without it. With it, falling behind becomes
smooth lag, and tracking jumps (measured up to a 192mm instantaneous
jump) get clamped to one frame's budget automatically.

Critical: the rate-limit origin must be the robot's actual current pose
(FK of `q_now`), not a freely-accumulating commanded value. The first
version used the latter: when IK got rejected the commanded pose kept
advancing anyway, diverging further from the actual robot configuration
→ cascading rejections → measured success rate barely moved (5.88% →
5.85%). Restarting from the actual pose every frame makes the rate
limiter self-correcting: rejections don't accumulate, and the robot
catches up once the hand stops.

### `dt<=0` handling

When `dt<=0` (first frame, or a timing glitch), the motion budget must be
0 (no motion allowed), not "unlimited." An earlier version wrote
`if dist > max_mm > 0:` — when `max_mm=0` the chained comparison is
`False`, falling through to `cmd_p = p_des` and bypassing the rate limit
entirely. No time elapsed should mean no displacement; this is now
written as an explicit branch instead of a chained comparison. The same
mistake pattern is guarded against again in `safety.py`'s `step_budget`.

### `arm_angle` ordering

Must be decided after the target pose is finalized: the nullspace
controller needs to probe "how good is the configuration this arm angle
solves to," which requires a target to solve against. The joystick only
contributes a bias term.

## tool_frame.py

### The problem it fixes

A recorded UMI pose is the **leader gripper's TCP: the two-finger midpoint**
(xense-taccap-lerobot `taccap_gripper/ee_transform.py`; the mount transform
is measured from CAD on both sides and was checked in Rerun on 2026-08-02 —
the EE marker sits on the finger midpoint). This stack's FK returns the
**flange**: `config.KINE_CFG`'s DH chain ends at the flange face and nothing
here calls `set_tool()`.

Hand a fingertip trajectory to an IK that targets the flange and the *flange*
faithfully traces the demonstrator's fingertips, while the robot's own
fingertips trace a path |t| further out. Nobody notices under pure
translation — a constant offset cancels in a relative mapping — and that is
exactly why the UMI replay looked fine without this. It only shows up under
**rotation**, which is where a manipulation demo spends its interesting
frames.

Measured on `replay_test`, arm A, `SCALE=0.5`, offline `replay.py --umi`:

| tool offset | clamped frames | Cartesian limiter |
|---|---|---|
| none (flange) | 15 / 2484 | 7 frames |
| `--tip 0,0,180` | **123 / 2484** | 65 frames |

Same episode, same budget, 8× the clamping — the flange has to swing through
an arc the fingertip never made. `scripts/umi_filter.py` reports the same
thing as a peak joint-rate ratio: 1.42× budget at the flange, 1.86× with a
180mm tip.

### Only the translation matters

A tool frame is (R_ft, t_ft). Replace R_ft with R_ft·A for any constant
rotation A and every commanded *flange* pose comes out bit-identical:

```
tip:   R_t' = R_f·R_ft·A            p_t' = p_f + R_f·t_ft   (unchanged)
map:   R_des' = dR·R_t0' = R_des·A  p_des' = p_des
back:  R_f' = R_des'·R_ft'ᵀ = R_des·A·Aᵀ·R_ftᵀ = R_f
       p_f' = p_des − R_f'·t_ft = p_f
```

The Cartesian rate limiter is invariant too — it only ever uses the *angle*
of `R_des·R_nowᵀ`, and the A's cancel there as well.

That is what makes a **pivot calibration sufficient**: it recovers `t_ft` and
can say nothing about `R_ft` (the touched point is rotationally symmetric),
and `R_ft` does not matter here. `scripts/tip_calib.py` therefore emits
`kine_offset` with a/b/c zeroed. The rotation is still carried in the
6-vector because the same entry feeds `Marvin_Robot.set_tool()`, whose
Cartesian targeting *does* need it.

The claim is self-tested rather than asserted (`python3 -m algos.tool_frame`):
two tool frames with the same translation and wildly different rotations are
run through the real `Retargeter` and must produce identical targets to
5.7e-14 — plus a control showing that a run with *no* tool differs, so the
first test cannot pass vacuously.

### Where it is wired in, and where it deliberately is not

`Retargeter(tool=…)` makes the tip — not the flange — the thing that latches
at clutch-press, that the Cartesian limiter bounds, and that `cart_lag_mm`
measures; only the final `target_xyzabc` is converted back to a flange pose
for the IK. `tool=None` is the default and is bit-identical to the
pre-`tool_frame.py` behaviour (verified: `replay.py --umi` reports the same
15 clamped frames / 0.9mm lag as before).

Passed on the UMI paths (`replay.py --umi`, `scripts/umi_replay.py`,
`scripts/umi_filter.py`). **Not** passed by live PICO teleop: the operator's
hand pose has no fingertip correspondence to enforce, and a human watching
the gripper closes that loop far better than a constant offset would.

### Status of the number itself

`configs/tool/umi.yaml`'s `kine_offset` is **all zeros = not measured**, for
both arms. `ToolFrame.load()` returns `None` for that rather than treating a
zero-length tool as a measurement, and every UMI entry point prints a warning
saying which trajectory it is actually judging. Measure it with
`scripts/tip_calib.py`; `kine_offset_source` is then required by
`drivers.tool_config`'s validator, same rule as `source`.

One thing a pivot cannot settle: *where along the finger* the recorded TCP
sits. The CAD "EE frame" is the authority; a few mm of axial disagreement can
survive this calibration.

## safety.py

### Design principle

Pure functions/small stateless-except-noted classes, unit-testable
without hardware or the SDK. Every "should this frame be sent" decision
belongs here, not scattered across `arm_driver`. Reject is safer than
allow — every function defaults to "don't pass" when uncertain.

### `clamp_joint_step` — isotropic scaling, not per-axis clamping

A clamp, not a rejection: small overshoot means the operator moved fast,
slowing down is enough; large jumps (solution-branch jumps) should
already be rejected at the `ik_solver` layer.

Must scale all 7 joints by the same factor, not clamp each axis
independently — the 7-vector IK solution determines the end-effector
direction jointly; clamping each axis to its own excess changes the
ratio between them, and the end effector moves in a direction IK never
actually solved for. Example (budget 0.202°):

```
IK says   J1 +0.05  J2 +0.10  J4 +0.80   ratio 1:2:16
per-axis  J1 +0.05  J2 +0.10  J4 +0.202  ratio 1:2:4   <- direction changed
isotropic J1 +0.013 J2 +0.025 J4 +0.202  ratio 1:2:16  <- direction preserved
```

Measured (arm following in engaged state, over frames that actually got
clamped): per-axis clamping produced a median direction deviation of
3.6°, p95 12.2°, max 16.1°, while 59% of frames were being clamped at
the time. Isotropic scaling brought that to 0.

Trade-off: axes that weren't over budget also slow down, so clamped
frames move a bit less overall. That's the right trade — IK just
validated this direction to 0.01mm tolerance; the last step shouldn't
undo it.

### `SafetyGate` — rate, not a per-frame constant

The step limit is a RATE (deg/s), not "degrees per frame." Per-frame
budget = rate × dt.

Previously this took a hardcoded per-frame constant (0.6°), which had
two problems:
1. Dimensionally inconsistent with the Cartesian rate limiter elsewhere
   in the same pipeline (`retarget.py`, which was already × dt).
2. Implicitly assumed the loop runs at a constant `CONTROL_HZ` — once it
   overran, the actual permitted joint speed dropped along with it, even
   though the robot still had the same distance to cover.

Switching to a rate means "how far this frame may move" is determined by
real elapsed time; loop jitter no longer affects end-effector speed.

The rate must be strictly tighter than what the controller can actually
execute (this is what config's speed-budget section derives). Looser
than the controller ⇒ `q_cmd` runs ahead of `q_meas` ⇒ tracking error
accumulates monotonically ⇒ guaranteed latch.

### `step_budget` — `dt<=0` ⇒ 0

Not "unlimited" — no elapsed time means no motion should happen.
(`retarget.py` had the same class of bug written as
`dist > max_mm > 0`, which took the unlimited branch exactly when
`max_mm=0` — same mistake, guarded against in both places now.)

### `check()` — tracking error: sustained vs. transient

Excess is accumulated as duration first: the caller treats a transient
excess as "just skip this frame"; only `sustained=True` should be
treated as a real fault. The criterion is duration, not instantaneous
magnitude.

## embodiment.py

### What it answers

"The human did this; which parts of it can *this* robot do?" The hand-held
rig has no joint limits, no velocity budget and a wrist that goes where a
7-DoF arm cannot follow, so some fraction of every UMI episode is not
reproducible on the Tianji. This walks the episode through the **real**
pipeline — `Retargeter` → `ArmIK.solve_with_backoff` → `SafetyGate`, the same
objects `ArmChannel` wires up — labels every control period, and cuts the
episode into spans to keep and spans to drop. CLI:
`scripts/umi_filter.py`. Runs offline: no robot, no network.

Reusing the pipeline rather than re-deriving a feasibility check is the whole
point. A filter that models the robot separately can disagree with the
pipeline that will actually run, and then a "kept" span still fails on
hardware. `_make_channel()` exists so the wiring lives in exactly one place.

### Three things that make a verdict conditional

1. **Feasibility is not a property of the demo alone.** Relative mapping
   reproduces displacement, not clearance or reach — the same 200mm sweep is
   comfortable from one start configuration and off the workspace edge from
   another. Every verdict is "…from `q0`", default `config.UMI_START_JOINTS`
   because that is what `umi_replay.py --goto-start` puts the arm at.
2. **The tool frame has to be set** (`tool_frame.py` above), or the filter is
   judging a flange trajectory that will never be commanded.
3. **"Too fast" is not "impossible."** The speed budget scales with `--speed`
   and `VEL_RATIO`; the workspace does not. `speed` and `ik:*`/`gate:*` are
   separate codes because only the first is fixed by slowing down, and
   `--sweep` turns that into a table.

### Per-frame codes

| code | meaning |
|---|---|
| `ok` | followed inside every budget |
| `slow` | limiter clamped, tip still within `max_lag` — degraded, faithful path |
| `speed` | over budget long enough that the tip fell behind (`lag > max_lag_mm` or `rot_lag > max_rot_lag_deg`) |
| `ik:<reason>` | `ArmIK`'s own rejection, after backoff and axis projection |
| `gate:<reason>` | `SafetyGate` blocked it (`joint_limit`, …) |
| `keepout:<axis>` | left `config.UMI_REPLAY_KEEPOUT` — checked on the flange **and** the tip |
| `broken` | after a fault, with `--no-resync` |

`ok`/`slow` are feasible, everything else is not. The threshold between
`slow` and `speed` is deliberately a *lag*, not "was it clamped": the
`replay_test` episode clamps 10–15 scattered frames at 1.0x while never
falling more than 1.8mm behind, and cutting those would shred a demo that
hardware replayed fine (`umi与天机replay.md` §4.3). Sustained over-budget
motion, by contrast, accumulates lag monotonically — at `--scale 2.5` the
same episode reaches 150mm.

The required joint rate is recorded **before** the gate clamps it — the
clamped value never exceeds the budget by construction, so it cannot answer
"by how much was this over budget."

### Faults and re-anchoring

Mirrors `ArmChannel` exactly, or the filter would be predicting a pipeline
that does not exist: a single rejection holds the last command and usually
recovers on the next frame (that is what `solve_with_backoff` is for), while
`MAX_CONSEC_REJECT` in a row — or any `gate:joint_limit` — is a fault.

On a fault the scan re-anchors: a fresh `Retargeter` latches at the next
frame onto wherever the arm now is, and scanning continues. That is the data
curation question ("cut the bad span, keep the rest"). `resync=False` stops
at the first fault instead, answering the other one: how much of this episode
survives in a single continuous take.

### Segmentation: bridge first, then drop short

`segment(verdicts, min_len, bridge)` swallows *interior* infeasible runs
shorter than `bridge` (default 50ms) into the surrounding span, then demotes
feasible spans shorter than `min_len` (default 0.5s). The order is
load-bearing: drop-then-bridge would turn one long span with a single hiccup
into two short spans and throw both away. Bridged frames are counted per
segment and printed — a bridge hides a frame the robot did not reproduce, so
it must not be silent. Runs at either end are never bridged: there is no
feasible span on both sides for them to belong to.

Segments carry **recorded** (30Hz) frame indices alongside control-period
ones, found by searching the raw timestamps rather than dividing by fps — a
dropped recording frame would silently shift every index the arithmetic way.
Ranges round inward, so a kept span never claims a recorded frame that is
only partly inside it. Those indices are what `--frames A:B` takes in
`replay.py --umi` and `scripts/umi_replay.py`, and what a dataset consumer
indexes by.

### `intersect()` — a bimanual episode is only usable where both arms can follow

One arm stalling mid-reach makes the *other* arm's frames unusable too: the
recorded action for that instant no longer describes what the robot did.

### Measured on `replay_test`

From `UMI_START_JOINTS`, `--scale 1.0`, `VEL_RATIO=55`, no tool frame:

| | arm A | arm B |
|---|---|---|
| feasible | 2484/2484 | 2484/2484 |
| peak joint rate | **1.42× budget** | **0.86× budget** |
| peak tip lag | 1.8 mm | 1.3 mm |
| worst limit margin | 27.4° | 24.9° |

Those ratios reproduce the on-robot measurement in `umi与天机replay.md` §4.3
independently: arm A needs ~127°/s against an 89°/s budget (1.42×) and got
10 frames clamped on hardware; arm B peaked at 76.6°/s (0.86×) and got zero.
The offline filter and the live run agree on which arm is the problem and by
how much.

From `HOME_JOINTS` instead, the same episode collapses — arm A keeps 8.8% of
its frames, all cuts `keepout:Z` — which is the 2026-08-25 near-collision
showing up as a filter verdict rather than as an operator hitting e-stop.

### What it does not model

No self-collision, same as everywhere else in this stack. `keepout` re-uses
`config.UMI_REPLAY_KEEPOUT`, a provisional 100mm Z floor from one observed
near-miss, not a measured body envelope — `--no-keepout` turns it off, and
the printed advice says so rather than letting a provisional number read as a
measurement. Tracking error is not modelled either: with `q_meas = q_cmd`
(perfect servo tracking) it is identically zero, the same assumption
`replay.py`'s offline loop documents.
