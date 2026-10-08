# Human scores and chunk seam metrics

Read the current metric definitions in the user's selected study register first.
This checkout maintains its register in `docs/experiments.md`. For another study
without a register, use the user's agreed definitions and identify missing choices.
Do not turn the examples below into frozen numeric settings.

## Human

Read `evaluation/human-label.json` in each selected finalized episode:

- `task_result`: `success`, `failure`, or `invalid`;
- `reviewer_id`, `review_mode`, `label_schema`, `created_at`;
- `failure_tags`, `operator_note`.

New labels use `human-label-v2` without a smoothness score. Historical v1 task
results remain usable; ignore their old `smoothness_score` and do not rewrite
raw labels.

Missing means unreviewed. Do not generate or overwrite a human assessment.
`result.json.success` describes runtime finalization; a normally ended rollout
can have a human task failure. Conversely, a runtime interruption does not
silently supply an absent human label.

For a rubric-compatible cohort, count `S` successes and `F` failures and report
`SR = S / (S + F)` as `S/N (%)`. Exclude invalid/unreviewed from this denominator
and report them separately. If `N=0`, SR is missing, not zero. Keep live and
blinded video reviews separate unless the protocol explicitly combines them.
Malformed labels are data-quality failures, not task failures.

## Seam calculation

For a reviewed boundary between outgoing plan `o` and incoming plan `i`, let
`t_i` be `committed.start_time_ns` of the incoming plan:

```text
q_old = interpolate outgoing committed joint plan at t_i
q_new = first incoming committed joint row
J_arm_mm = 1000 * norm(FK_arm(q_new).xyz - FK_arm(q_old).xyz)
episode_mean_arm = mean(J_arm_mm for eligible boundaries in this episode)
episode_p95_arm = percentile(eligible J_arm_mm, 95, method="linear")
episode_max_arm = max(eligible J_arm_mm)
setting_mean_arm = mean(episode_mean_arm for eligible episodes)
setting_p95_arm = mean(episode_p95_arm for eligible episodes)
setting_max_arm = max(episode_max_arm for eligible episodes)
```

Use the recorded embodiment/tool FK and action units, and keep left/right separate.
P95 denotes the 95th percentile, not a confidence interval. The register uses the
mean of per-episode P95 values; do not replace it with a pooled-handoff percentile.
The Max column keeps the worst eligible seam across the cohort, not the mean of
per-episode maxima. Use the same eligible boundary set for all three statistics.
Save each arm's worst seam episode and boundary identity. Main result tables use
six separate columns in mm: L-Mean, L-P95, L-Max, R-Mean, R-P95, R-Max.
Do not combine arms into one cell or average them together.
This is a **committed reference seam**, not measured displacement or command jump.
For a continuous seam require actual outgoing/incoming execution evidence and
valid outgoing coverage at the boundary. Serial end-hold restart comparisons are
a different boundary class; use the task's explicit hold convention and report
them separately. Do not silently clamp expired trajectories into a continuous
seam. No eligible boundaries means missing.

Match recorded `request_seq`/plan IDs, committed timestamps, `ticks.plan_id` and
control times; inspect `events.jsonl` and any restart/hold evidence. Exclude startup,
Home/Resume restart, superseded plans that never execute, and cross-segment pairs.
If the stored evidence cannot establish eligibility, retain an unknown reason.
Save reviewed boundary identities and their class alongside the numeric output.

## Existing plotter: exploratory diagnostics

From the repo root, using an environment with the offline FK dependencies:

```bash
"$MM_ANALYSIS_PY" scripts/validation/plot_chunk_handoffs.py \
  --episode "$MM_EPISODE" \
  --robot-config "$MM_ROBOT_CONFIG" \
  --out "$MM_NEW_ANALYSIS_DIR" \
  --label "$MM_UNIQUE_CASE_ID"
```

Resolve these variables from the selected run. Always pass the robot config:
the default is Tianji. Current supported layouts are dual-arm YAM
(`manimux/configs/embodiment/robot/yam_dual.yaml`) and Tianji-TacCap
(`manimux/configs/embodiment/robot/tianji_taccap.yaml`); `left_arm/right_arm` are
hardcoded, so do not claim arbitrary-embodiment support. `RobotModel.from_config`
uses offline FK, not the hardware runtime. A known local analysis environment is
`envs/yam/.venv/bin/python`; verify availability/dependencies before use.

Outputs are PNGs, overview sheets, a PDF, `handoff-summary.csv` and JSON.
The CSV contains `old_chunk`, `new_chunk`, request IDs,
`left_position_seam_mm`, and `right_position_seam_mm`. It does not contain a
validated eligibility decision or aggregate Mean/P95/Max. `--max-handoffs N` truncates
the analysis and must not be used for a complete formal result.

The plotter pairs adjacent **accepted** plans and clamps the old sample to its
endpoint; it does not implement the boundary review above. Nonzero
`source_offset_steps` also affects its source-timeline illustration. Treat its
CSV as candidates; calculate summary statistics only over the explicitly
reviewed pairs. A missing pause/takeover record may prevent formal scoring even
when a picture can be drawn. There is currently no complete, validated unified
seam evaluator to invoke instead.

Do not run historical `data/analysis/.../generate.py` as a general CLI: some bind
one episode/output and assert a fixed handoff count. Existing image-only subsets
or old settings cannot be promoted to complete current experiments.
