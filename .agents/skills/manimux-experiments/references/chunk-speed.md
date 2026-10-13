# Chunk speed diagnostics

Generate a self-contained HTML report to compare early and late motion within
action chunks. The report includes scheduled references, executor commands,
measured feedback FK, saved decoded trajectories, per-chunk heatmaps and timelines.
An optional Pi05 probe adds native offline predictions, demonstration speeds,
paired wrist images and inference latency.

The report generator runs offline. It does not connect to a robot or camera,
command motion, or use the running inference server.

## Recorded rollouts

Run from the repository root in the ManiMux environment, with the local robot FK
dependencies installed:

```bash
.venv/bin/python scripts/validation/plot_chunk_speed.py \
  --session data/experiments/EXPERIMENT/session-ID \
  --out data/analysis/chunk-speed
```

Open `data/analysis/chunk-speed/chunk-speed.html` directly in Firefox. No HTTP
server, JavaScript package installation or network access is required. All chart
data and optional JPEG previews are embedded in the HTML. The directory also
contains `analysis.json`, `summary.json` and, when supplied, `offline-summary.json`.

The input contract is a finalized ManiMux session with `left_arm` and
`right_arm`. Joint and gripper dimensions come from the recorded assembly:
YAM uses six arm joints plus one gripper; Tianji may use seven plus one.
Each rollout must contain `meta.json`, `events.jsonl`, `result.json`
and `data.zarr` with accepted `canonical_raw`/`committed` plans and all three tick
streams (`scheduled`, `command`, `state`). Partial rollouts and those without
usable plan ownership are excluded. Repeated disjoint runs of one plan ID are
rejected instead of being joined across pauses.

The generator verifies the recorded hashes of repository embodiment sources and
configs before FK. Use the matching checkout when they differ; it deliberately
does not silently recompute historical evidence with changed geometry. Absolute
recorded embodiment config paths are relocated to this checkout. External SDK
binaries and private model assets are not covered by that source check. Input
files are read only, and the report output must be outside the session directory.

The `ziyang` integration imports the report from upstream commit `c387397` and
uses assembly-owned arm dimensions rather than a fixed seven-joint slice. The
existing YAM environment can run the recorded-rollout report:

```bash
envs/yam/.venv/bin/python scripts/validation/plot_chunk_speed.py \
  --session /absolute/path/to/session-ID \
  --out data/analysis/chunk-speed/session-ID
```

Existing recording is sufficient for this diagnostic: tick `monotonic_ns`,
`plan_id`, `scheduled`, `command`, `state`, saved plan arrays and their `dt_ns`,
plus session configuration/source hashes. No extra speed array needs to be logged
during control. `command` is the executor target submitted to the robot interface,
not a follower receipt or the CAN-sent snapshot. `state` speeds are estimates from
recorded feedback positions on the control-tick clock, not motor-reported velocity.
The recorder does not preserve each feedback packet's original acquisition time;
duplicate cached feedback can create alternating zero/spike estimates. The included
resampling sensitivity view checks this effect but cannot recover absent timestamps.
YAM's current state timestamp/sequence describe host reads, not motor update events.

For the experiment, retain the three speed-phase curves, per-arm late/early
ratio and fraction of eligible chunks with ratio <= 0.8. Freeze thresholds and
show eligible/excluded counts. Aggregate per rollout before comparing methods;
the report's pooled chunk median is a session diagnostic, not an equal-rollout
experiment statistic. Slowing down can be appropriate near contact and is not
automatically failure or worse smoothness. Native model-output diagnosis would
add the pre-adapter action array, action layout/dt, request ID and conditioning
metadata; `canonical_raw` is already decoded and cannot substitute for it.

The optional offline probe below remains specific to Tianji pack-plate 20D data
and its model contract; it is not a YAM coin/bottle collector.

## Optional upstream native predictions

The report can attach completed upstream Pi05 pack-plate probe output with
`--offline /path/to/probe-output`. It validates checkpoint/normalization pairing,
completion metadata and speed-array shapes before displaying it. The collector
is not imported here: this branch lacks the upstream pack-plate model contract
and recipe, and they are unrelated to YAM coin/bottle evaluation. See the
[upstream collector](https://github.com/SII-LiuLab/manimux/blob/c38739783011f38b2ad6d7f7144e965914ac1e8b/scripts/validation/pi05_pack_plate_chunk_speed.py)
for that separate workflow. Offline predictions are not executed robot motion.

## Reading the comparison

TCP translation speed is adjacent FK distance divided by actual elapsed time, in
mm/s. Joint L2 and rotational speeds are also recorded in `analysis.json`, but are
not mixed into the displayed TCP metric. Gripper motion is excluded from joint L2.

Each execution window is one consecutive nonempty tick `plan_id` run. Speed
statistics never difference across plan boundaries. Startup, final censored runs,
missing adjacent plan ownership, fewer than 15 ticks and gaps exceeding 50 ms
are excluded. The first/last thirds are **time-weighted**, including fractional
overlap of tick intervals. The default eligibility threshold is mean TCP speed
at least 5 mm/s with early speed greater than 1 mm/s. The report takes the median
of per-chunk late/early ratios; a ratio at most 0.8 means at least 20% slowdown.
Profile bands show chunk quartiles, not confidence intervals.

The saved `canonical_raw` stage has already passed through the action adapter;
its length can differ from the model horizon. It cannot establish the shape of
the original native prediction. Offline native speeds use 31 adjacent intervals
from 32 predicted actions, excluding observation-to-first-action displacement.
GT uses the corresponding future states. The optional first-12-interval view
compares a shorter horizon; it does not reproduce online cropping or conditioning.
Offline timeline emphasis ends at the next sampling interval and is only a visual
guide, not evidence that those predictions were executed.

Inference latency includes model preprocessing, sampling and synchronized output
conversion, excludes video decoding/preview encoding, and excludes the first JIT
call from its steady-state summary. The front/back comparison is descriptive:
chunks in one rollout are correlated, and these plots do not identify a causal
model, RTC, IK or executor fault.

Report code lives in `scripts/validation/plot_chunk_speed.py` with frontend assets
in `scripts/validation/chunk_speed_assets/`. Generated reports, previews, datasets
and native action archives belong under ignored `data/analysis/`, not in Git.
