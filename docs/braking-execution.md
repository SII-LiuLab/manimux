# Braking-aware joint tracking and bounded execution prefixes

`configs/sapolicy/yam/infra/manimux-braking-h25.yaml` enables the optional
`execution.smooth.tracking_mode: braking` mode. Existing profiles retain the
legacy mode unless explicitly changed. MPC is unchanged. The bottle profile also
uses the optional [latched release](independent-ik-execution.md) rule.

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

`execution.max_chunk_steps: 25` limits the ordinary joint timeline to original
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

Validation on 2026-09-09:

- 63 executor, timeline, configuration and runtime regression tests passed.
- A fixed 0.1 rad target at 0.8/3 formerly reached 0.1587 rad. The new mode
  converges without crossing the target while respecting velocity/acceleration.
- Tests cover moving references, plan replacement, unavoidable reversal,
  position-bound braking, inference-gap stopping and 50-to-25 prefix handling.
- On the recorded raised-speed SA episode, baseline reconstruction error was
  zero. Right-arm reference-to-command EEF error P50/P95 changed from 61.6/211.9
  mm to 47.4/143.7 mm. Left-arm P50 improved from 23.8 to 10.9 mm; P95 increased
  slightly from 60.5 to 62.7 mm. This is fixed-input offline replay, not a new
  model rollout or physical tracking validation. Gripper output can differ at
  inference gaps because the new mode holds the existing command.

Local evidence: `/home/ubuntu/sa/diagnostics/braking_executor_20260909/`.
