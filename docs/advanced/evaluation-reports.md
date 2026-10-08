# Per-task HTML evaluation reports

The report entry point reads recorded attempts, saved human labels and existing
trajectory metrics. It produces one self-contained HTML report per task, with
SVG plots, comparison tables, attempt details and downloadable data. It does not
run robots, policy inference or PRM inference.

The report opens with one wide Model / Method comparison table per task containing
Human, left/right seam and all 11 PRM columns. An optional study register
can supply the planned comparison rows.
Handoff and 10/20/30-step similarity plots follow the table. Metric definitions,
provenance, attempt details and downloads are collapsed below.

Run from the repository root using the existing YAM environment:

```bash
envs/yam/.venv/bin/python -m manimux.evaluation \
  --task-dir data/evaluation_tasks/put_bottles
```

Pass multiple task directories after `--task-dir` to produce separate reports.
The command prints the absolute HTML path for every completed task. Subsequent
runs publish a new version; they never overwrite a previous report.
Each task has its own table; no cross-task aggregate table is generated.

## Task index

Each selected directory contains `task.yaml`. Paths resolve relative to that
file, including `output_root`, `raw_roots`, `episode_dir` and `overlap_metrics`.
A task index is independent of the runtime scoring rubric in
`manimux/configs/evaluation/put_bottles.yaml`.

```yaml
schema: task-report-v1
task_id: put_bottles
title: Put bottles into the bin
output_root: ../../analysis
experiment_table:
  path: ../../../docs/experiments.md  # Optional study register.
  heading: Table 1. 抓瓶子放入箱子 · YAM · 首轮主任务
  bindings:
    Pi05 / Serial: explicit-checkpoint-and-setting-version
groups:
  - setting_id: explicit-checkpoint-and-setting-version
    model: Pi05 Joint 30k
    method: Serial
    split: dev
    target_attempts: 30
    expected:
      meta.task: Put the bottles into the bin.
      meta.runtime: serial
      meta.policy_backend.model.checkpoint_variant: pi05_yam_put_bottles_joint_step_30000
      manifest.config.inference.algorithm: serial
    episodes:
      - episode_dir: ../../experiments/example/rtc/session-ID/rollout-001
        overlap_metrics: ../../analysis/example/overlap-metrics.json
        handoff_summary: ../../analysis/example/handoff-summary.json
        handoff_images:
          - chunk-02-to-03-eef-xyz.png
```

`groups: []` produces a pending report. A group with `episodes: []` records a
planned setting without inventing attempts or scores. Real inputs require an
explicit `expected.meta.task` mapping: folder names do not establish task identity.
Add exact dotted fields from `meta`, `manifest` or `result` to bind a checkpoint,
method and configuration. Matching those fields does not certify the complete
experimental protocol. Keep historical data separate from new aligned settings.

Optional `experiment_table` reads the exact Markdown heading's model/method rows
on each generation, expanding `+ RTC` etc. under their parent model. Its source
hash and selected cells are saved. `bindings` explicitly maps a `Model / Method`
row to a setting ID for measured Human results; unbound historical settings remain
separate rows. No scores are matched using a model name alone. Tasks absent from
the register display their configured groups or a pending row.

Optional `raw_roots` discover `session-*/rollout-*` or direct `rollout-*` children.
Explicit episode entries attach metric artifacts and override discovery entries
for the same path. Duplicate paths within a setting count once; mapping an episode
to two settings is an error. Separate attempts with the same layout/repeat remain
visible and are not silently selected or discarded. Roots and all episode inputs
must be outside the report output tree.

Optional `handoff_summary` and `handoff_images` attach selected existing plotter
PNGs to an episode. Each filename must match exactly one summary row. The report
copies and embeds the original image and records image/summary hashes. This is an
explicit task-index attachment, not automatic episode identity verification or
formal seam qualification; image captions are candidate diagnostics only.

## Output contract

With the local indexes under `data/evaluation_tasks/<task>/`, output is:

```text
/home/ubuntu/manimux/data/analysis/<task>/
  latest.json
  reports/<UTC-timestamp>-<unique-id>/
    index.html
    summary.json
    episode-metrics.json
    episodes.csv
    cohort.jsonl
    provenance.json
    task.yaml
    figures/
      overlap-position-rmse.svg
      overlap-velocity-cosine.svg
      handoff-<attempt-id>-<selected-image>.png
```

The two figure files exist only when numeric data is available. The HTML embeds
all plotted SVGs, selected PNGs and its report data; it works offline without CDN assets. Keep
the entire directory when sharing its companion downloads. HTML controls support
attempt filtering, full-report JSON export and browser printing to PDF.

Publication uses a staging directory and atomic rename. `latest.json` is updated
only after the report bundle completes. It contains absolute `html` and `summary`
paths. Source hashes and expected identity checks are saved with the report.
The raw episodes, human labels and input metric artifacts are never modified.
Local task indexes and generated artifacts live under ignored `data/`.

## Statistical scope

- Human SR uses saved success/failure labels, separately for live and video review.
  Invalid, missing, excluded and malformed labels have separate counts. Runtime
  completion is not task success. Wilson intervals describe attempt-level SR;
  they do not implement matched-layout block confidence intervals.
- Count-based human labels add H-Score as mean `completed_count / target_count`.
  The saved rubric must match the episode metadata. Different rubrics have separate
  score summaries. Legacy binary labels do not acquire synthetic bottle counts.
- The trajectory importer accepts complete `overlap-metrics.json` artifacts from
  `scripts/validation/plot_chunk_handoffs.py`. It checks episode identity, input
  metadata hashes, metric schema and numeric fields. Different metric definitions,
  windows, action intervals, FK configuration hashes or implementation hashes within
  a setting are excluded from aggregate curves.
- At each window, use the source's common valid boundary set across all requested
  windows, then average per-episode means with equal episode weights. Left/right
  arms remain separate. Cosine has its own denominator because stationary motion
  has no defined direction. Per-window counts and reasons remain in the episode
  data. Missing metrics are never converted to zeros.
- These are committed-reference candidate diagnostics, not measured robot motion
  or certified executed handoffs. Formal seam qualification and matched PRM case
  import remain separate unfinished steps. No formal seam/PRM scores are invented.
  Import does not recompute Zarr arrays or establish historical FK equivalence.

The initial local report inventory covers `put_bottles`, `coin`, `stack_blocks`,
`stack_blocks_small`, `assemble_screwdriver`, `pass_ball` and `redball`. Only the
explicit historical bottle episode is currently selected as report evidence;
other tasks await an explicit evaluation cohort. Training demonstrations are not
automatically included as evaluation trials.
