# Evidence, progress and report updates

## Locate evidence

Start from the selected experiment's `run.output_dir` and the register's evidence
links, including its output-path register. Do not assume a universal `data/prm/` directory or scan all videos/weights.
Use `rg --files --hidden --no-ignore <selected-output-directory>` with metadata
filename globs to discover the relevant campaign: experiment data is commonly
gitignored. Scope this override to the selected data directory, not the whole
checkout or weight tree.
Resolve paths from the context of the saved configuration and session location.

```text
<output_dir>/session-*/
  session-manifest.json
  rollout-NNN/                       # finalized attempt
    meta.json
    result.json
    events.jsonl
    data.zarr/
      ticks/{monotonic_ns,plan_id,inference_ms,camera_time_ns,...}
      ticks/{state,scheduled,command}/<group>
      plans/NNNNNN/{canonical_raw,infra_output,committed}/<group>
    videos/{index.json,<camera>.mp4}
    evaluation/human-label.json       # only if a human saved a label
  rollout-NNN.partial/                # unfinished, retain and count separately
```

Discover optional files; do not assume they exist because the directory exists.
Read Zarr schemas/attributes before loading arrays; some finalized runs have zero
accepted plans. Never scan/import a robot driver to inspect these files.

Sources: `manimux/recording/episode.py`, `manimux/recording/video.py`,
`manimux/evaluation/manual.py`, `manimux/cli.py::_create_run_dir`,
`manimux/runtime/edge.py::run`, and `manimux/session.py`.

## Identify the actual run

- Manifest `config` is the resolved snapshot. `config_sha256` hashes the entry
  YAML bytes, **not** that snapshot or all referenced files. A new analysis may
  hash a canonical serialization of `config`, recording its convention, but must
  not relabel the existing hash.
- Manifest git SHAs do not capture dirty source, dependencies or weight content.
  Preserve available provenance and name missing identities rather than implying
  complete reproduction.
- Rollout `meta.json` contains task/layout, `experiment_mode`, backend identity,
  adapter, action timing, etc. New experiment rollouts freeze `layout_id`, `repeat_id`
  and `reference_layout={task,path,sha256}` at Prepare. The gallery task is not the
  canonical task ID or policy prompt; map it explicitly in the evaluation protocol.
  Serve-mode task/identity can differ from the original session snapshot; use rollout
  metadata for these per-attempt fields. Legacy missing identities remain unknown.
- `runtime=manimux` alone does not identify Serial. Check the snapshot's
  `inference.algorithm` **and** `inference.inference_schedule`, then policy and
  checkpoint identity. Do not group by robogui label or output folder alone.
- Identify an episode by its full resolved directory, keeping both session and
  rollout names. Different sessions reuse `rollout-001`; runtime `session_id` and
  directory/manifest IDs need not be the same identifier.
- The current recorder does not persist a setting ID/version or full block/seed
  identity. Explicitly map an episode to a setting using the snapshot and backend
  evidence. If the comparison cannot be established, leave it unassigned.

## Record progress without inventing results

Use separate counts: attempts, finalized, partial, human-valid, human-invalid,
unlabeled, seam-valid, PRM-valid, and processing failures. These counts need not
agree. Read the per-task/model/method target and counting basis from the register.
Do not split one row's budget across other models or methods. A target range is
not a frozen count; a missing exact target means the completion fraction is unknown.
Keep failed attempts and do not choose extra trials after inspecting scores.

Distinguish these stages: planned; configured; collected; partially evaluated;
evaluated; reported. A model file or a server-ready log does not establish
collection or task success. Historical runs can be inventoried as historical
without filling a current formal result row.

The experiment register's progress section is the durable status record; if an
older register lacks it, add a small section for the selected task/setting. Update
counts together with the setting version, last analysis time, selected cohort,
next missing item and evidence path. Do not overwrite old finalized results when
the setting changes: retain their version and add the new comparison.

## Derived analysis and Markdown

Create a unique analysis directory, normally under ignored
`data/analysis/<task>/<setting>/<split>/<analysis-id>/`. Avoid the plotter's default
output name, which can collide across sessions. Raw recordings and human labels
remain unchanged during scoring.

For each analysis, save `cohort.jsonl`, `provenance.json` and
`episode-metrics.json` alongside reused tool outputs. Follow the register's
per-attempt index fields: exact resolved episode/session paths, host and split,
raw artifact paths, human label, seam output, PRM manifest/run/case identity, and
per-metric eligibility or missing reasons. Index every attempt, not just successes.
Use null for absent outputs; never invent a path merely from a naming convention.
Record the actual directories returned by the PRM toolkit, not an assumed run name.
Link the register's path row, progress row and per-task table context to this index;
a root directory or wildcard alone is not a cohort. Keep previous analysis versions.
These are this skill's derived-output conventions,
not files that the current recorder already creates. Include:

- selected episode paths, identity-to-setting mapping, dev/test and block/layout;
- source files/digests, analysis code version and exact commands/parameters;
- per-metric included/excluded episodes or boundaries, with reasons;
- per-episode human values, reviewed seam statistics and matched PRM case IDs;
- all selected table metrics, their per-metric denominators (including conditional
  PRM subsets), metric/judge versions and interval method.

Calculate from explicit eligible sets; use standard tools such as Python's
`statistics.mean` for finite values. For scientific intervals, use the method in
the register and retain its seed/block membership. Do not pool all handoffs from
all episodes, silently use complete cases across all metrics, or replace missing
values by zero.

Before changing Markdown, match the exact task table and setting/version, including
the parent model for indented algorithm rows. Result tables contain only model /
method names and metrics: update shared settings, sample counts and evidence in
the task context and path/progress register outside the table. Do not reintroduce
N_task or Setting columns. Update the result cells, evidence and progress together. Never run a broad model-name replacement across all
tables. If a metric definition remains proposed, report exploratory results as
such; do not fill a formal column as though its protocol were frozen.

## Recording limitations that affect eligibility

Viser displays runtime data; `EpisodeRecorder` persists trajectories/videos.
Viser also writes human labels and separate layout-reference artifacts. A field
visible in the GUI is not necessarily saved.

- Ticks are recorded in RUNNING, not through all pause/home states. There is no
  complete timestamped pause/resume segment log or dedicated actual-plan-takeover
  event. `plan_boundary` describes acceptance. Use tick plan IDs and timing to
  qualify a boundary, and mark unresolved intervals unknown.
- `ticks.monotonic_ns` is the control timestamp; original robot-state sample
  timestamp/sequence and camera sequence are not persisted. Video capture times
  are stored, but do not guarantee exposure-level synchronization.
- `canonical_raw` means decoded canonical actions, not original model tensors.
  Full request observations and rejected action arrays are not retained.
- `ticks.inference_ms` repeats the latest accepted request's value. Averaging all
  ticks does not produce mean latency per inference request.
- New formal rollouts save reference path/hash and repeat; they do not copy or
  prevent overwriting reference PNGs, nor record randomized block identity. Ordinary
  rollouts clear formal identity. Human labels have one mutable sidecar, not reviewer history.
- New recordings omit `ticks/optimized`: it duplicated `command` exactly. Old
  files may contain it. Keep the three plan stages and camera timestamp indexes;
  their meanings/sampling rates differ and they are not redundant.
