---
name: manimux-result-analysis
description: Analyze saved ManiMux rollouts and collection episodes, including chunk latency, handoffs, command tracking, gripper timing and video quality. Use recorded evidence without starting hardware or treating predicted actions as measured task success.
---

# ManiMux Result Analysis

Paths below are relative to the ManiMux root. Start with the user's episode or explicit
comparison set. If they ask for the latest run, locate it under the selected config's
`run.output_dir`; filesystem modification time alone does not identify the intended task.
Keep original recordings unchanged and put derived plots/clips in a separate local location.

## Find the recording format

Runtime episodes are written by `manimux/recording/episode.py`:

| Artifact | What it contains |
|---|---|
| `meta.json` | Recorded runtime/policy metadata; inspect actual keys and backend identity |
| `events.jsonl` | Episode, inference and plan-boundary events emitted by that runtime |
| `result.json` | Runtime termination and recording summary; not automatically a human task label |
| `data.zarr/ticks` | `monotonic_ns`, `plan_id`, `inference_ms`, per-group `state`, `scheduled`, `optimized`, `command`, and `camera_time_ns` |
| `data.zarr/plans` | Per-plan `canonical_raw`, `infra_output`, `committed` stages with timing attributes |
| `videos/index.json` and camera MP4s | Written-frame timestamps, configured FPS and dropped-bundle summary, when video is enabled |

Interrupted episodes may remain in `.partial` directories and lack finalized arrays.
Legacy demonstration recordings use a separate format. Inspect the offline readers in
`manimux/viewer/replay_data/` rather than assuming runtime Zarr paths. These recordings can include
`manimux-control.jsonl`, controller targets, measured joints and separate timestamps.

## Keep the evidence stages separate

- `canonical_raw` is the adapter's canonical action chunk, not necessarily the model's
  original normalized tensor. Compare it with `infra_output` and `committed` before
  attributing a change to the model or the scheduler.
- Tick `scheduled` / `optimized` / `command` are planned or issued values; `state`
  is robot feedback. Use command-versus-state traces to discuss tracking or closure delay.
- Read that body's group layout and gripper convention from metadata/config. A closing
  target or a red Viewer block is not proof of contact or a successful grasp.
- Runtime `result.json.success` can indicate successful completion of the loop. For task
  success, use the experiment's human label or clearly identified observational evidence.

## Time, chunks and video

Use recorded time units: runtime monotonic nanoseconds, `inference_ms` milliseconds,
and each plan's `dt_ns`. Collection may contain Unix nanoseconds and camera milliseconds;
inspect its schema before mixing them. Do not assume tick index equals action index.

For latency statistics, count each inference once using request/plan identity or events;
tick records can repeat the same inference value. Separate model inference duration from
request-to-handoff delay. For RTC/manimux, derive trim, old-tail execution and conditioning
from the selected strategy and recorded metadata, not orange cell counts or one universal
latency-rounding rule. Identify which plan supplies a condition and which consumes it.

Check the actual sensor path when aligning frames: the ordinary camera-server driver
currently stamps receipt time, while the UMI_DP/Tianji sensor preserves server frame times.
Those server times are host capture/callback times, not hardware exposure synchronization.
Prefer the video index timestamps over `frame_number / fps` for robot alignment. For an
external screen recording, inspect its actual stream FPS, resolution and trimming offsets;
do not assume it shares the runtime clock or silently resample a requested original-rate clip.

For plan-boundary comparisons, reuse `scripts/validation/analyze_chunk_boundaries.py`
with explicit `--episode` arguments. Its automatic discovery is not a universal index
of newer session layouts. Run it with an existing environment containing its dependencies.

Answer the user's concrete question with episode paths, the relevant time window and a
small number of measurements or plots. Separate observations from proposed causes; note
missing frames or fields rather than inventing them. For success rates, state the included
episodes and labels. Do not change controller parameters as part of a read-only analysis.
