# Human scores and chunk seam metrics

Read the current metric definitions in the user's selected study register first.
This checkout maintains its register in `experiments.md`. For another study
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
`result.json.run_completed` describes runtime finalization; a normally ended rollout
can have a human task failure. Conversely, a runtime interruption does not
silently supply an absent human label.

For a rubric-compatible cohort, count `S` successes and `F` failures and report
`SR = S / (S + F)` as `S/N (%)`. Exclude invalid/unreviewed from this denominator
and report them separately. If `N=0`, SR is missing, not zero. Keep live and
blinded video reviews separate unless the protocol explicitly combines them.
Malformed labels are data-quality failures, not task failures.

For configured partial credit, validate the saved `evaluation` against the
episode's `meta.json.evaluation`. Count rubrics derive `score` from
`completed_count / target_count`; stage rubrics derive it from the selected
`stage_id` option. A stage score is a total, never a sum of reached milestones.
Use the recorded rubric, not today's YAML, and reject inconsistent labels.
Report H-Score as the mean valid score alongside SR; invalid/unreviewed labels
have no score, and binary-only labels do not supply a missing partial score.
Keep differing rubric snapshots separate. The task report exposes both count
and stage summaries in `scores_by_rubric`. Rule setup and examples are in
[the study guide](../../../guideline.html#experiments).

## Native-action seam calculation

For a study evaluating the model-plus-inference action-sequence join, use original
model action rows after required coordinate/action decoding. Do not temporally
interpolate either endpoint. This is the manuscript profile adopted on 2026-10-29;
older same-time reference results remain a separately named diagnostic.

For each execution-qualified outgoing/incoming pair:

```text
j_old = min(floor((last_old_tick_ns - old_start_ns) / old_dt_ns), old_retained_length - 1)
raw_offset = infra_source_offset - raw_source_offset + time_trimmed_steps + handoff_skipped_steps
q_old = old_canonical_raw[old_raw_offset + j_old]
q_new = new_canonical_raw[new_raw_offset]
J_arm_mm = 1000 * norm(FK_arm(q_new).xyz - FK_arm(q_old).xyz)
episode_mean_arm = mean(eligible J_arm_mm)
episode_p95_arm = percentile(eligible J_arm_mm, 95, method="linear")
episode_max_arm = max(eligible J_arm_mm)
setting_mean_arm = mean(episode_mean_arm for eligible episodes)
setting_p95_arm = mean(episode_p95_arm for eligible episodes)
setting_max_arm = max(episode_max_arm for eligible episodes)
```

Require the old tick to be at or after its plan start. Verify matching action
space, observation origin and action interval, and that the entire retained
committed prefix matches its mapped raw slice (absolute tolerance 1e-8, zero
relative tolerance). Reject a mismatch rather than substituting blended,
interpolated, measured or command endpoints. The retained prefix length limits
the outgoing index; the unexecuted tail is not an executed row.

Use the recorded embodiment/tool FK and action units. Keep six separate columns
in mm: L-Mean, L-P95, L-Max, R-Mean, R-P95, R-Max. Do not average arms together.
P95 is a quantile, not a confidence interval; setting P95 averages rollout P95s,
not pooled boundaries. Setting Max is the worst eligible boundary, not an average
of rollout maxima. Save its rollout and boundary ID. All three statistics use
the same eligible boundary set. No values means missing.

Match actual tick ownership, request/plan IDs and boundary events. Require the
switch to be bracketed by outgoing/incoming ticks; exclude startup, pauses,
Home/reset/restart, long gaps and plans never observed executing. Keep continuous
and end-hold classes separate. Constant recorded commands establish end-hold
eligibility, but the numerical endpoint is the last reached original action.
Do not label an expired-plan boundary continuous by clamping its endpoint.

This is an **action-sequence join distance**, including ordinary between-step
motion. It is not a same-time trajectory difference, physical displacement or
issued-command jump. Timing and latency still select the retained rows. There is
no requirement that it be smaller than the interpolated-reference metric.
The last native timestamp strictly before takeover is a separate sensitivity;
do not substitute it silently for the last observed outgoing execution tick.

### Current implementation and historical results

The ordinary `manimux/evaluation/seams.py` and plotter still calculate the legacy
interpolated committed-reference metric. The manuscript update did not migrate
them. Do not relabel their output as native-action seams.

The local frozen implementation and validation are in
`data/analysis/native-action-seams-20261029/`. Its cohort-specific `recompute.py`
reuses execution-eligibility rules and maps endpoints back to original rows.
Do not use the frozen-cohort script as a general CLI without explicitly adapting
the cohort and FK profile. Current paper values and definitions are in
`/home/ubuntu/manimux-overleaf/source/native-action-seams-20261029.json` and
`native-action-seams.md`; `scripts/apply_native_action_seams.py` reapplies only
the six seam columns after historical importers. Other metric definitions and
original source snapshots are retained.

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
when a picture can be drawn. The task report now invokes `manimux/evaluation/seams.py` with execution
evidence from `execution.py`: observed tick ownership, matching request/plan
boundary events, bracketing control times and pause/reset exclusions. Never-executed
accepted plans are not paired. The report separates `continuous` and `end_hold`,
retains rejected boundary identities/reasons and exports each arm’s worst case.

Do not run historical `data/analysis/.../generate.py` as a general CLI: some bind
one episode/output and assert a fixed handoff count. Existing image-only subsets
or old settings cannot be promoted to complete current experiments.
