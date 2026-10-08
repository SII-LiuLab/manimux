# Execution and action decoding

Executors generate commands from timeline references. Action decoding converts model
actions into those references; it is a separate stage. The options below are specialized
choices, not requirements for a basic joint-space deployment.

## Decode in a separate process

`manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_manimux_direct_async.yaml` preserves SAPolicy's
full-pose IK and direct joint execution. It enables `policy.action_decoding:
process`; the default remains `inline` for existing configurations.

Pair it with `manimux/configs/policy/sapolicy/yam/teleop50-raw.yaml`: both now use a
50-step horizon at 30 Hz (about 1.67 seconds). The server returns all 50 model
steps and the adapter decodes all 50. Rolling replanning still removes expired
prefixes and replaces the remaining trajectory when a new plan arrives; it does
not guarantee open-loop execution of every row. Direct execution still has no
chunk-boundary blending, so increasing the horizon does not remove command jumps.

The adapter converts EEF targets into complete joint-position trajectories in
two spawned processes, one per arm. Each owns its adapter, MuJoCo data and IK
solver. Within an arm, targets retain their original order, previous-successful
solution seeds, iteration budget, tolerances, clipping and failed-step fallback.
The measured state at decode submission seeds both arms. Warmup initializes the
solvers before robot connection, without connecting hardware.

The control loop continues sampling and sending the existing trajectory while
decoding runs. It publishes a new plan only after both arms finish and their
contracts agree. If the previous trajectory expires, the existing runtime hold
path applies. No partial-arm plan is published. The inference-plus-decoding
pipeline permits only one in-flight request, so slow IK cannot build an unbounded
backlog. A dead or deadline-exceeding decoder faults the runtime and uses its
ordinary robot shutdown path.

This adopts the execution separation used by UMI's
[independent interpolation controller](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/franka_interpolation_controller.py)
and checks actual receipt time before accepting actions, as illustrated by its
[environment scheduling](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/bimanual_umi_env.py).
It is not UMI's robot controller or a replacement with differential IK.

### Time and pause semantics

Decoding completion and commit timestamps use the current clock. Plans beyond
their original final source timestamp are rejected, including the final-step
edge case where integer trimming alone would resurrect an expired last point.
Partially available plans use the existing joint timeline's trimming and
interpolation semantics; this does not introduce Cartesian interpolation or new
joint smoothing. It also does not implement calibrated camera/actuator latency
compensation. Faster preparation changes which source rows remain available,
so identical closed-loop hardware motion is not guaranteed by identical IK output.

By default the measured state at submission seeds IK. The arm keeps following
the previous plan while decoding runs, so on a fast arm that seed lies behind
the arm when the new plan is committed. `inference.expected_decode_s` (process
decoding only, default 0) sets the expected submit-to-commit time. The seed is
then the active plan's reference at `now + expected_decode_s`
(or at its end if it finishes earlier), and the adapter's execution time moves
by the same amount, skipping rows before it. Without an active reference the
measurement remains the seed. If decoding finishes earlier than expected, the new
plan can start up to that difference early. `decode_submitted` records
`seed_source`, `seed_time_ns` and `expected_start_ns`.

Process decoding stops requesting inference while paused, clears its active
trajectory and invalidates outstanding responses. Resume needs a fresh observation;
home similarly invalidates results from before the home operation. Pausing does
not synchronously wait for or interrupt an individual numerical solve. Finish
closes the robot and decoder processes; their cleanup has bounded waits.

### Adapter contract

This mode requires a strategy named `manimux` or `rtc`: a built-in strategy, or
a plugin that delegates to one, such as the UMI measured-history plugin. An adapter
must declare `supports_context_only_decode = True`: its decode result must be
fully determined by raw action plus `ActionContext`, without observation-side
mutable caches. Optional `decode_partitions` and `decode_action_partition`
partition independent groups; optional `warmup_decode` initializes a child.
Other adapters retain inline decoding unless they explicitly support this contract.

SAPolicy observation preprocessing retains its own separate FK model. Decoding
never holds that model's lock. The parent sends only actions and a copied state
snapshot; robot drivers, images and robot control sockets are not passed to the
decoder. Startup has no model RPC or motor commands.

### Diagnostics and verification

Recorded plans include `raw_model_eef` in the original model frame and per-arm
`ik` diagnostics: seed, each solve's convergence and time, and failed-step count.
`decode_ms` is the maximum partition compute time; `decode_stage_ms` includes
communication and polling delay. `observation_to_commit_ms` includes the full
observation/model/decode/commit path. Neither compute metric is a control frequency
or a task-success measure.

```bash
envs/yam/.venv/bin/python scripts/validation/benchmark_parallel_ik.py \
  /path/to/saved/direct/diagnostics --output /tmp/parallel-ik
```

This consumes `offline-ik-profile.npz`, `offline-ik-profile.json` and `signals.npz`
from the existing recorded-observation diagnostic. It performs no robot I/O or
model-server calls. Every output joint and failed-solve status is compared against
serial full IK. IPC can outweigh the parallel speedup for already-cheap targets;
the most reliable benefit is removing full IK from the control thread.

Integration tests also inject 250 ms of CPU-bound decoding while a mock runtime
continues issuing commands, check expiry and pause invalidation, exercise failed
IK equivalence and reject incomplete/invalid dual-arm results. These tests do not
establish hard realtime scheduling or successful hardware grasping.

## Independent group IK and tool sequencing

The SAPolicy braking profile enables `inference.independent_group_decoding` for
independent bottle manipulation. Coupled tasks retain the default atomic decode
failure behavior. The model still predicts 50 source rows; `max_chunk_policy_steps: 25`
is now passed into decode, so the SA adapter skips expired rows and rows 25–49
before IK. `source_offset_steps` prevents a second latency trim. All 50 absolute
EEF wire predictions remain in `raw_model_eef` for diagnosis.

Each arm runs in its existing decoder process. `decode_budget_ms: 40` bounds the
cooperative IK work for one arm's executable prefix. The Mink objective, pose
tolerances and model joint bounds are unchanged. The bounded solver exits on
convergence, deadline, iteration cap or 12 iterations without meaningful residual
improvement (relative threshold 1e-4). It takes at least one update before checking convergence, matching i2rt and avoiding a tolerance-sized deadband for small moving targets. It verifies FK residuals after joint clipping.
A local solve failure does not establish global unreachability.

At the first failed row, the adapter stops solving that arm's remaining rows and
marks `ActionChunk.hold_from_step`. The successful prefix remains available. The
timeline adjusts this index for trimming and requests a hold before interpolation
or velocity feedforward can enter an invalid segment. Smooth braking decelerates
that arm from its last commanded velocity using the existing acceleration limit;
its gripper immediately holds its last command. The other arm continues. A new
feasible chunk clears the hold and resumes from the existing executor state.
Holding is not instantaneous stopping or a classifier for task completion.

The process client stops waiting after the IK budget plus 40 ms of scheduling/IPC
allowance (or the request deadline, whichever is earlier). It publishes a typed
hold for a missing arm together with the available arm. A still-busy worker gets
no additional job until its old result is drained; stale results cannot replace
newer plans. Native QP calls cannot be interrupted cooperatively, but a hung QP
does not block publication for the other arm. Unexpected worker crashes or malformed
actions retain error handling. This is a best-effort timing budget, not a hard
real-time guarantee on Linux.

After the September 10 speed feedback, the profile uses arm limits 0.6 rad/s and
1.5 rad/s², with continuous gripper limits
1/s and 12/s², 0.2 s refill threshold, and GUI Start control. There is no workspace
sphere projection, accepted high-residual solution, semantic idle-arm detection,
or change to model weights. The relative model actions are each decoded against
the same observation pose, not cumulatively integrated within the chunk.

Validation artifacts: `/home/ubuntu/sa/diagnostics/independent_ik_20260909/`.
Recorded rollout-002 replay: 72 chunks, process decode P95 730.2 → 16.5 ms,
maximum 742.6 → 17.9 ms; left arm held in 0 chunks, right arm in 32. Replay
used recorded joint seeds and the first source gripper target as seed aperture;
it did not send hardware commands or verify task success.

### Bounded orientation recovery

The bottle profile now opts into `policy.adapter.kinematics_options.recovery_ori_threshold`
(5 degrees) and `recovery_max_joint_delta` (0.35 rad per waypoint relative to its seed).
Strict full-pose IK remains the first attempt. On failure, orientation costs 0.1 and
0.02 are tried within the same arm budget. Acceptance still requires position error
at most 4 mm, model joint limits, bounded orientation error and bounded joint change.
Recovery is recorded as `recovered_orientation` with the original strict result.
If these attempts fail, a final recovery starts from the original waypoint seed
with joint-position constraints intersecting the physical limits and seed ±0.35
rad. This searches for a nearby solution instead of only rejecting distant ones.
The final candidate is clipped to those bounds and revalidated against the same
4 mm / 5 degree tolerances. The extra attempts share the existing decode deadline;
successful earlier attempts return unchanged. Diagnostics identify this fallback
as `recovery_method: seed_box`. Recorded last-bottle regression evidence is in
`/home/ubuntu/sa/diagnostics/last_bottle_ik_20260910/`.
It changes the allowed orientation error; it does not establish collision-free motion.
Unacceptable candidates still produce a per-arm braking hold.

Offline replay artifacts are in `/home/ubuntu/sa/diagnostics/right_recovery_20260910/`.
For the latest 83 recorded chunks, right-arm holds decreased from 42 to 5; none of
those five started at the first decoded row. For the previous 105 chunks, holds
decreased from 63 to 2, including one rejected by the recovery joint-change bound.
Decode P95 was about 37 ms on this host. Full-pose residuals and raw 50-row targets
remain recorded; execution stays capped at the first 25 original source rows.
A delayed ideal follower test covers recovered movement, command speed/acceleration
bounds, and release only after measured position catches up. No offline result
establishes real-robot grasp or placement success.

### Finish a release before accepting a replacement target

The bottle profile selects `release_guard.mode: latched_release`. This feature is
opt-in; profiles without `release_guard` retain their existing gripper behavior.
With latched release, a commanded closed
gripper (aperture at most 0.35) arms a release event. When the unblended reference
requests aperture at least 0.85, the executor captures that waypoint's joint pose.
It approaches this fixed pose under the same braking limits, holds the gripper
until measured FK is within 20 mm, then commands full opening. Later chunks cannot
replace this pose or close the gripper during the event. An IK hold in a newer
plan does not invalidate the already accepted release pose.

Completion requires commanded aperture at least 0.98 and measured aperture at
least 0.95. The arm then holds the release pose and open gripper until a plan based
on an observation captured after completion arrives. This prevents a stale close
prediction from reversing the release. Ordinary inference gaps continue the event;
explicit Pause, Home and reset cancel it. The other arm follows its own reference.
The executor records phase, captured pose, originating plan and measured error as
JSON-compatible `gripper_decision` events, including during inference gaps.

Release approach and opening each have a `release_guard.phase_timeout_s` deadline
(2 seconds in this profile). On timeout, the executor records failure, brakes the
affected arm, and preserves its current aperture until a valid plan using a
post-failure observation arrives. It then resumes ordinary model tracking and
bypasses further release latches for that arm until Pause/Home/reset. Timeout
alone does not force the gripper open or mark placement complete. Grasp latches
and the other arm's release latches keep their own state. This closes the same
unbounded-wait failure mode as the grasp timeout below. Tests and recorded-feedback
replay are under `/home/ubuntu/sa/diagnostics/release_timeout_20260910/`.

Model execution remains capped at 25 original source rows per chunk. Finishing a
release can extend the time spent at a captured waypoint; it does not execute the
discarded source rows. Arm limits remain 0.6 rad/s and 1.5 rad/s², and gripper limits
remain 1/s and 12/s². This is not a bin detector or proof of successful placement.

Validation: `/home/ubuntu/sa/diagnostics/latched_release_20260910/`. Replaying the
failed episode with recorded feedback reproduces the old commands exactly. Using
the recorded references with an ideal delayed follower after the first captured
release, the new executor reaches full opening without exceeding arm limits.
Unit and mock-runtime tests cover replacement close predictions, inference gaps,
the fresh-observation barrier, cancellation and JSON recording. After the live
trial the operator reported correct behavior with imperfect grasp alignment.
This is a working deployment baseline, not a measured grasp-success rate.

### Finish grasp closure before lifting

The bottle profile also enables `executor.smooth.grasp_guard`, sharing the
release guard's FK model and event scheduling. After the gripper has been open,
the first unblended reference below the existing open threshold (0.85) captures
the closure-onset pose. Capturing a later fully-closed waypoint would risk using
a pose already on the lifting trajectory.

The executor holds the aperture while approaching that fixed pose. Closing starts
when measured and commanded FK are both within 20 mm and 5 degrees of the pose,
and commanded arm speed is at most 0.1 rad/s. It then commands the configured
closed aperture with the existing gripper speed/acceleration limits. New chunks
cannot move the arm away or reverse closure during this event.

The bottle profile sets `grasp_guard.approach_max_velocity: 0.25` rad/s. This
per-arm cap applies before a grasp is latched, while the commanded gripper is
above its closed threshold, and throughout the grasp approach/closing phases.
It uses gripper command state as a coarse approach cue, not object proximity.
The arm also retracts more slowly with an open gripper. Closed-gripper transport
and release events retain the normal 0.6 rad/s cap. Acceleration remains 1.5
rad/s², so entering the slower mode decelerates continuously; gripper shaping
keeps its independent limits. Profiles omitting this option retain normal speed.
Recorded and ideal-follower evidence is under
`/home/ubuntu/sa/diagnostics/grasp_precision_20260910/`; it does not establish
object localization accuracy or successful collision-free grasping.

Completion requires the close command to reach its endpoint, continued pose
arrival, and a measured aperture below 0.80 that stays within 0.01 for 150 ms.
A held bottle can leave a nonzero aperture: measured zero is not required. This
is a motion-completion check, not proof of object contact or a successful grasp.
The arm continues holding until a plan using a post-completion observation arrives.
Ordinary inference gaps preserve the event; Pause/Home/reset cancel it. The other
arm remains independent. Grasp phases and the captured target are recorded in
`gripper_decision` alongside the existing release diagnostics.

Approach and closing each have a `grasp_guard.phase_timeout_s` deadline (3 seconds
in this profile, allowing the slower approach to settle). A timeout is recorded
as failure, not grasp completion. The arm
brakes and preserves its current aperture until a valid plan based on an
observation after the failure arrives. It then resumes ordinary tracking and
bypasses further grasp latches for that arm until Pause/Home/reset. The other
arm's grasp guard and both arms' release guards remain active. This prevents a
persistent tracking error from locking the executor to an old grasp waypoint.
Regression evidence is under
`/home/ubuntu/sa/diagnostics/grasp_timeout_20260910/`; recorded-feedback replay
does not establish physical grasp success after recovery.

The 25-row source prefix and arm/gripper limits are unchanged. Offline evidence
is under `/home/ubuntu/sa/diagnostics/grasp_guard_20260910/`: exact reproduction of
the failed rollout with this feature disabled, then a delayed ideal follower with
nonzero aperture at simulated contact. Physical grasp alignment is not established
by that replay.

## Braking-aware tracking

`manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_manimux_braking_h25.yaml` enables the optional
`executor.smooth.tracking_mode: braking` mode. Existing profiles retain the
legacy mode unless explicitly changed. MPC is unchanged. The bottle profile also
uses the optional [latched release](execution.md) rule.

The tracker combines the reference velocity from adjacent control samples with
position-error correction. Correction speed is capped by a stopping-distance
envelope that includes one control tick. Absolute joint-position boundaries also
have a stopping envelope. Velocity changes obey the configured acceleration
limit; command velocity is retained across chunk replacements. A target that
changes inside the current stopping distance can still be crossed. This is an
acceleration-limited tracker, not a jerk-limited or collision-aware planner.

During a RUNNING inference gap, this mode decelerates the existing arm command
to rest instead of resetting it to lagging measured feedback. It holds the last
gripper command, except that a pending release continues toward its captured pose
and completes opening. Explicit Pause, Finish and Home retain their existing behavior.
The fixed control period is used for command differences, as in the existing
executor; controller overruns and motor tracking must still be monitored.

`inference.max_chunk_policy_steps: 25` limits the ordinary joint timeline to original
source indices 0 through 24 **before** latency trimming. Model inference and
recorded raw model chunks remain 50 rows; the decoder solves only the usable prefix.
A three-row trim therefore commits
22 rows, not 25 additional rows. Responses whose entire permitted prefix is
already obsolete are rejected. The next request is submitted with less than
0.2 seconds remaining; an earlier response can replace the current chunk before
all 25 original rows execute. The policy's diffusion sampling count is unchanged.

This SA profile keeps full IK, RAW weights, 30 Hz action spacing, 100 Hz control,
arm limits 0.6 rad/s and 1.5 rad/s², and continuous gripper limits 1 and 12. In braking
mode `cutoff_hz` determines the position-correction gain; it is not a cascaded
low-pass filter. The bottle profile captures both grasp and release events, and
uses the measured-FK arrival and gripper-completion rules documented above.
