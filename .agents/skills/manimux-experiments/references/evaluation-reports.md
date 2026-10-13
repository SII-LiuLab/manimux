# Per-task HTML evaluation reports

The report entry point reads recorded attempts, saved human labels and existing
trajectory metrics. It produces one self-contained HTML report per task, with
SVG plots, comparison tables, attempt details and downloadable data. It does not
run robots, policy inference or PRM inference.

The report opens with one wide Method comparison table per task containing
Human, completion time, NLDLJ, NSPARC, left/right seam and all 11 PRM columns. An optional study register
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
  path: ../../../experiments.md  # Optional private study register.
  heading: Table 4. Bottles · YAM
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

## Completion time and smoothness

These three metrics complement success rate and boundary discontinuities. Keep
overlap RMSE as an existing diagnostic; no additional RMSE main-table column is
required. Smoothness alone cannot establish task success or physical tracking quality.

**Time(s)** is an explicit human annotation from the first formal Start to the
instant the full task success criterion is met. In RoboGUI, enter it in
`Completion seconds from Start (optional)` when saving a successful evaluation.
Leave it blank when unknown. For video review, the same writer accepts
`completion_time_s=...` and `review_mode="video"`. The saved v2 human label adds:

```json
{"completion_time": {"seconds": 12.3, "origin": "first_formal_start", "endpoint": "task_success", "source": "human_annotation"}}
```

Finish time and `result.wall_time_s` are not task completion timestamps. Failed,
invalid and unlabeled attempts do not acquire completion times. The report keeps
live/video timings separate, gives the valid n and stores sample standard error
in `summary.json` (null for n < 2). Compare time together with success rate:
success-conditioned time can favor a method that only completes easy layouts.
Pauses remain part of elapsed Start-to-success time; interrupted attempts should
be marked invalid or kept in a separate setting, not compared to uninterrupted trials.

Enable offline smoothness with one task-wide profile in `task.yaml`:

```yaml
smoothness:
  robot_config: ../../../manimux/configs/embodiment/robot/yam_dual.yaml
  groups: [left_arm, right_arm]
  sample_hz: 30
  max_gap_s: 0.1
  cutoff_hz: 10
  threshold: 0.05
```

The robot config is an explicitly selected offline FK assembly; it does not
connect hardware. Select the assembly/tool matching the recording. Source paths
resolve relative to the task file. Re-run the usual report command; no separate
metric service or raw-record modification is needed.

The `scheduled-eef-smoothness-v1` profile reads `ticks/scheduled`, the timeline
reference before executor filtering, not raw model candidates, SDK/CAN commands
or measured `state`. It linearly resamples recorded joint references on a common
30 Hz grid, then uses the assembly FK to obtain each EEF position and orientation.
The window spans the first through last recorded RUNNING tick, including ordinary
holds. It is not automatically cropped at the human success timestamp. Use the
same stop protocol across methods. A pause/Home event inside the window or a tick
gap above the configured threshold makes the metric unavailable; the implementation
does not interpolate through such a break or pick a favorable segment.

- **NLDLJ ↓**: `ln(T³ * integral(||jerk||² dt) / peak_speed²)`. Translation uses
  finite differences of XYZ; rotation uses spatial SO(3) increments. Four-pose
  jerk stencils that cross plan changes or ambiguous cross-plan interpolation
  samples are excluded. Duration and peak speed refer to the full window.
  Retained/excluded stencil counts are exported. A separate seam metric still
  matters because this measure excludes the connection itself.
- **NSPARC ↓**: positive spectral arc length of linear/angular speed magnitude,
  FFT padding level 4, adaptive band ending at the last amplitude >= 0.05 within
  10 Hz. It retains the full speed sequence, including switches. No extra smoothing
  or detrending is applied. These finite-window metrics depend on the task and
  stop protocol; compare matched tasks, rates and profiles.
- Each episode averages active EEF translation/rotation channels equally. Stationary
  channels are missing, not zero scores; undefined logarithms/insufficient samples
  also remain missing. A metric with an undefined active channel is not averaged
  over just the remaining channels. Setting summaries require identical active
  channel sets and profile signatures, then weight episodes equally. Both successful
  and failed eligible attempts contribute; human-invalid attempts do not. Channel
  membership and component values are retained for review before comparing settings.

The velocity-normalized T³ definition follows the
[smoothness reference implementation](https://github.com/siva82kb/smoothness/blob/master/python/smoothness.py).
[Legato](https://arxiv.org/pdf/2602.12978) prints T⁵ with peak-speed normalization;
this implementation deliberately uses the standard dimensionless T³ form and is
not an exact numeric reproduction of its table. Scheduled references and resampling
also differ from a metric applied directly to unexecuted model output chunks.

The main table and `episodes.csv` include Time(s), NLDLJ and NSPARC. The report adds
`figures/outcome-time.svg`, `outcome-nldlj.svg`, `outcome-nsparc.svg` when values exist:
each dot is an attempt and the vertical line is its setting mean. Expanded attempt
details and `episode-metrics.json` retain missing reasons, profile, FK/component
source hashes and a digest of input ticks, references and evaluated poses. Referenced
external model assets are not copied into the report; archive the assembly dependencies
with the experiment for reproducibility. No timing labels are inferred for old episodes.

## Statistical scope

- Human SR uses saved success/failure labels, separately for live and video review.
  Invalid, missing, excluded and malformed labels have separate counts. Runtime
  completion is not task success. Wilson intervals describe attempt-level SR;
  they do not implement matched-layout block confidence intervals.
- Count-based human labels add H-Score as mean `completed_count / target_count`;
  stage-based labels use the selected option's total score.
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


## Configurable human scoring

Experiments reference a task rubric with `evaluation.config`. Users edit the YAML
before loading the experiment. RoboGUI displays the task name and a single result
selector after Finish; it does not contain a template editor. Layout/repeat setup
remains in `run.experiment_template`.

```yaml
evaluation:
  config: ../../../evaluation/coin.yaml
```

Supported types:

- `binary`: Completed / Not completed.
- `count`: `completed_count / target_count`. Block stacking uses target 2;
  bottles retain target 6.
- `stages`: select one named total-score option. Coin uses 0 / 0.3 / 1;
  completing a deposit scores 1, not 1.3.

```yaml
kind: stages
task_id: coin
rubric_id: coin_stages_v2
label: Coin transfer and deposit
options:
  - {id: incomplete, label: No coin picked up, score: 0, success: false}
  - {id: picked_up, label: Coin picked up by one hand, score: 0.1, success: false}
  - {id: handover, label: Handed over, score: 0.3, success: false}
  - {id: above_bank, label: Coin held above piggy bank, score: 0.5, success: false}
  - {id: deposited, label: Deposited in target, score: 1, success: true}
```

IDs and labels must be nonempty and unique. Scores must be finite numbers in
[0, 1]; only score 1 is complete success. Keep stage criteria explicit in labels
and version `rubric_id` when they change. Selection starts unlabelled: no result
is silently scored zero. Invalid attempts remain excluded and receive no score.

The runtime stores the loaded rubric in each episode's `meta.json`; human labels
save the selected `stage_id`, derived score, task result and rubric snapshot.
Editing the YAML requires reloading the experiment and does not alter old labels.
Reports verify the snapshot, keep different rubrics separate, and report H-Score
(mean total score, displayed as %) independently of H-SR (complete success rate).
Both count and stage summaries are in `scores_by_rubric`.

Current templates: `manimux/configs/evaluation/coin.yaml` and
`manimux/configs/evaluation/stack_blocks.yaml`. Their Pi05 task entries select
them explicitly. Small-block dataset variants and rotating-clock scoring remain
unbound until their task criteria are selected; prompt text never chooses a rubric.
