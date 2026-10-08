# PRM-as-a-Judge SOP

Use `scripts/evaluation/prm_as_a_judge.py`, which loads the checkout's
`PRM-as-a-Judge/eval/prm_judge/cli.py`. Read that CLI if options have changed.
Current subcommands are `eval`, `visualize`, and `serve`; there is no `prepare`
subcommand or automatic ManiMux rollout exporter.

## Select the judge profile and inputs

Read the selected judge profile in the experiment register/evidence, including
model revision, weight identity, evaluation mode, reference image, view mapping,
frame interval, normalization and postprocessing. If it is not fixed, prepare
the manifest and propose the missing profile; do not silently apply CLI defaults
and report a comparable formal score.

Write one JSONL case per eligible episode in a new analysis directory. Use a
unique session-plus-rollout case ID and a model key containing the setting ID;
otherwise Serial and RTC can be grouped together. Example schema, with illustrative
paths/IDs that must be replaced by actual verified inputs:

```json
{"case_id":"session-id_rollout-003","task_name":"task-id","task":"actual rollout instruction","video":"/absolute/main.mp4","left_wrist_video":"/absolute/left.mp4","right_wrist_video":"/absolute/right.mp4","goal_image":"/absolute/frozen-goal.png","label":"success","benchmark":"manimux_yam","model":"setting-id-version","metadata":{"episode_dir":"/absolute/session/rollout-003","setting_id":"setting-id-version","human_label_path":"/absolute/session/rollout-003/evaluation/human-label.json","video_index_path":"/absolute/session/rollout-003/videos/index.json","judge_profile_id":"profile-version"}}
```

- Map only actual human `success/failure` labels to `label`. Keep invalid and
  unreviewed episodes in a separate inventory. If scoring their progress as
  diagnostics, keep them separate from success-conditioned statistics.
- `video` is required; dual wrists must be supplied together. Do not mix
  `wrist_video` with the dual-wrist fields. Paths resolve relative to the manifest
  directory; absolute input paths avoid ambiguity.
- Read actual filenames from `videos/index.json`, not presumed camera names.
  Metadata paths are provenance: the evaluator does not read them to acquire
  human labels or synchronize images.
- Confirm every selected video's readability, frame count, timestamp count,
  common time coverage and cross-view skew. The current Dopamine adapter aligns
  by encoded frame index, truncating unequal lengths, not capture timestamps.
  If alignment fails, generate separately authorized synchronized derived inputs
  with a frame-source map, or mark the case ineligible. No automatic sync exporter
  exists today. Single-view input is repeated into three slots: label it single-view.
- Validate the exact goal file before inference. Without an explicit goal the
  CLI defaults to a blank image; if the effective reference does not exist the
  adapter can fall back to that rollout's last frame. Freeze an explicit goal or
  explicitly chosen blank reference and its digest to prevent accidental leakage.

## Commands and execution boundaries

Resolve `MM_PRM_PY`, `MM_MANIFEST`, `MM_CHECK_DIR`, and `MM_RUN_DIR` first; use new
output directories. A local interpreter candidate is
`/home/ubuntu/miniconda3/envs/prm-judge/bin/python`; a local weight candidate is
`checkpoints/pretrained/Robo-Dopamine-GRM-2.0-8B-Preview`. Verify these locally;
their presence does not establish GPU readiness or a particular model hash.

Manifest parsing only (writes preparation metadata, does not validate videos):

```bash
env -u PRM_GPU_MEMORY_UTILIZATION "$MM_PRM_PY" \
  scripts/evaluation/prm_as_a_judge.py eval \
  --manifest "$MM_MANIFEST" --run-root "$MM_CHECK_DIR" --dry-run
```

For authorized GPU evaluation, populate all following variables from the chosen
profile and verified resource assignment. These variables express the command
template; this skill does not choose their values.

```bash
CUDA_VISIBLE_DEVICES="$MM_GPU" PRM_GPU_MEMORY_UTILIZATION="$MM_GPU_FRACTION" \
  "$MM_PRM_PY" scripts/evaluation/prm_as_a_judge.py eval \
  --manifest "$MM_MANIFEST" --run-root "$MM_RUN_DIR" \
  --prm dopamine --prm-path "$MM_JUDGE_WEIGHTS" --gpus 0 \
  --eval-mode "$MM_EVAL_MODE" --frame-interval "$MM_FRAME_INTERVAL" \
  --batch-size "$MM_BATCH_SIZE" --normalize "$MM_NORMALIZATION" \
  --source-type "$MM_SOURCE_TYPE" --outlier-method "$MM_OUTLIER_METHOD" \
  --smoothing "$MM_SMOOTHING" --smoothing-weights "$MM_SMOOTHING_WEIGHTS" \
  --success-source label --keep-cache --visualize
```

Dopamine loads vLLM and weights inside this process; no separate policy server
is needed. Use one authorized visible GPU for this wrapper: its memory-budget
patch is not reapplied by upstream multi-GPU worker launches. Capture command,
environment overrides and logs as provenance, without recording credentials.

To recompute **existing genuine** curves, use `eval --prm recorded`, provide
`progress` or `progress_path` in the manifest, and unset
`PRM_GPU_MEMORY_UTILIZATION`. This is not new judge inference. Also unset that
variable for dry-run and visualization; setting it imports the GPU backend early.
`visualize --run-root ...` creates static reports. `serve` starts a process and is
separate from evaluation; do not launch it merely to read a result. A request to
write the skill or summarize existing scores does not authorize GPU inference.

## Validate outputs and fill the PRM columns

Read the chosen run's `run_params.json`, `discovery_manifest.json` and
`per_case.jsonl`. Each case has `status`, `progress_raw`, `progress_processed`,
`trace`, `postprocess` and `metrics`. Case directories contain
`result_summary.json` and, for Dopamine, `pred_vllm.json`. The toolkit also creates
`run_summary.json`, `report.md`, spreadsheets and visualization artifacts; it
does not automatically create an episode `evaluation/prm-v1.json` sidecar.

Verify exact case-to-episode membership and profile before using a score:

1. Require `status=ok` and a nonempty finite processed curve.
2. Check logs and raw `pred_vllm.json` predictions for parse errors. This adapter
   can replace an unparsable score with zero while still reporting `ok`.
3. Check `metrics.MP == max(progress_processed)` within numeric tolerance. Fill
   all PRM columns selected by the register, preserving its units and directions.
   Rate columns expressed as percentages need multiplication by 100; continuous
   columns remain dimensionless.
4. Filter to the exact task/setting/split cohort before aggregation. Use a manifest
   `model` key that distinguishes inference settings. Match the task and setting
   group in `run_summary.json.metrics.groups`; do not copy a cross-task leaderboard
   or Excel Model_Mean into a task row. Validate group membership against the index.
5. Use the toolkit's `aggregate_metrics`/`trace_summary` semantics with the recorded
   metric profile: ordinary per-case `metrics` does not contain FNS/SQS, and its
   threshold SR can differ from the effective SR when `success_source=label`.
   DRR aggregates only actual-drawdown cases, FNS only failures, SQS only successes.
   Save each subset's N; no applicable cases means a missing table cell, even if
   the toolkit returned zero. Do not average every conditional score over all cases.
6. Report `N_prm` and failed/missing cases separately. Do not remove PRM failures
   from the human SR denominator. Register the exact run directory, manifest,
   case IDs and report paths in the cohort index and experiment Markdown.

Human SR always comes from human labels. Even `--success-source label` falls
back to an MP threshold for missing/unrecognized labels; PRM SR is not a substitute.
Reject empty curves instead of using the toolkit's numeric zero fallback.

Frame interval counts encoded frames, not elapsed capture time. Current trace
timestamps do not give a verified chunk-event clock; do not overlay them as if
they were synchronized without a derived timestamp mapping. Preserve raw and
processed curves, all source IDs, and judge/goal/view/processing identities.
Use a new output run when a profile changes; `--skip-existing` only checks file
existence and does not prove matching settings.
