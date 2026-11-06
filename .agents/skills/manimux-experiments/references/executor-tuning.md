# Frozen-reference executor tuning

`scripts/validation/tune_umi_execution.py` screens legacy SmoothExecutor settings
on recorded UMI-DP Tianji pack-plate and pack-bottle trajectories. Run from the
repository root in the analysis environment:

```bash
.venv/bin/python scripts/validation/tune_umi_execution.py \
  --root data/experiments/umi-dp-bottle-plate-scratch-tf32-20261017 \
  --latest-per-task 20
```

The command creates a unique directory under `data/analysis/umi-executor-tuning/`.
`--output` can select a new directory; an existing directory is never overwritten.
It does not instantiate robot hardware, call a policy, or edit deployment YAML.

## Offline boundary

- Use frozen session configs, rollout task metadata and actual tick plan IDs.
  The alias `pack the plate` maps explicitly to `pack_plate`.
- Reproduce the recorded arm command with continuous executor state before
  scoring any intervention. The default maximum absolute replay error is 1e-6 rad.
  Failed replay episodes consume a cohort slot and remain visible as exclusions.
- Hypothetical commands feed the next executor tick, including across chunk
  boundaries. They do not replace subsequent IK seeds, observations or policy
  outputs. This isolates the executor's effect on the recorded references; it is
  not a closed-loop counterfactual rollout, a simulation of contacts, or evidence
  of robot/task success. Even chunk IK can change after a real intervention.
- Keep `handoff_skip_steps` fixed. It is a collection/deployment controller-gap
  calibration, not a disposable latency row. All inference settings, gripper
  behavior, control/action timing, velocity and position bounds stay recorded.
- Gripper commands are retained only for offline geometry; no candidate gripper
  feedback or contact behavior is fabricated. All primary metrics concern arms.
- Step IK depends on the previous command. Its recorded scheduled trajectory
  cannot be frozen to evaluate a different executor as a complete rollout.
  Braking tracking also needs lookahead that ticks do not fully record. These
  modes, feedback guards and per-group holds are explicitly excluded.
- Inference gaps keep the hypothetical last command and reset velocity, as in
  the runtime. Measured-state placeholders identify gaps, subject to the baseline
  replay gate. Long recording gaps and explicit pause/home/reset events reanchor
  once to recorded command. Derivative stencils never cross these anchors.

## Search and interpretation

The grid tests cutoff 4/8/12/20/40 Hz, retained acceleration or caps 8/16/32 rad/s²,
and isotropic/per-joint limits. Caps are `min(recorded, candidate)` when a recorded
limit exists; no recorded acceleration limit is relaxed. The recorded baseline
is an additional candidate. Duplicate effective settings are retained explicitly.

Selection uses older sessions for development and the latest third of sessions
for holdout. Never split chunks from one session between partitions. A single
session gives development evidence only. Partial attempts remain stress evidence,
not successful completed tasks. Summary strata retain checkpoint/backend, timing,
inference and recorded limiter identity. Cross-stratum means are descriptive;
a holdout with a new limiter setting is a transfer check, not an IID test.

Within each rollout, score joint tracking, speed clipping, nominal command
acceleration/jerk and actual plan-switch velocity jumps. Average rollouts equally,
then arms. Smoothness includes switches and inference holds. Derivatives use the
nominal executor period and describe commanded sequences, not measured robot
dynamics. A smaller clipping count alone is not the objective.

The conservative joint screen permits at most 5% worse tracking and 10% worse
acceleration/jerk/seam metrics on **each** development episode/arm, with no velocity
or position violations. These are analysis tolerances, not hardware limits.
Up to five best survivors undergo full-rate offline FK. All recorded embodiment
source/config hashes must match. TCP accuracy/rotation allow 5% and TCP speed,
acceleration and jerk 10% regressions per episode/arm. Select a survivor only if
joint RMS improves by at least 1%. Test the frozen selection on holdout; do not
select from holdout results. If none passes, retain the recorded baseline.

Separate development-average tradeoff choices show tracking-only, average-balance
and smoothness-priority options. They can fail individual trajectories or TCP
checks and are **not deployment recommendations**. Per-joint clipping changes
joint-motion direction and can improve joint RMS while worsening TCP error.

## Artifacts and verification

- `cohort.jsonl`: exact paths, inclusion/exclusion, final/partial status, source and
  config identities, baseline reproduction errors.
- `search-space.json`: frozen grid, locks, selection protocol and offline limits.
- `episode-metrics.json`: every candidate's per-arm metrics, split, FK shortlist
  and recorded tracking context. Missing geometry is explicitly excluded.
- `summary.json`: development/holdout scores, individual violations, strata,
  conservative selection and exploratory tradeoffs.
- `*-candidate-review.yaml`: review-only resolved settings for each recording;
  these are intentionally not executable experiment configs.
- `provenance.json`: invocation, source hashes, cohort exclusions and weighting.

Local `tests/unit/test_executor_tuning.py` checks the batched kernel against the
real SmoothExecutor across saturated chunks, reversals, holds and resets, verifies
parameter locks and session splits. Tests follow repository ignore policy.

Before promoting any candidate, collect new hardware trials that measure TCP
tracking, contact/grasp/release timing, acceleration/jerk and task outcomes. Do not
infer these from old measured states after changing hypothetical commands.
