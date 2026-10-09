# Chunk speed diagnostics

Generate a self-contained HTML report to compare early and late motion within
action chunks. The report includes scheduled references, executor commands,
measured feedback FK, saved decoded trajectories, per-chunk heatmaps and timelines.
An optional Pi05 probe adds native offline predictions, demonstration speeds,
paired wrist images and inference latency.

These are offline diagnostics. Neither command connects to a robot or camera,
commands motion, or uses the running inference server.

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

The current input contract is a finalized ManiMux session with `left_arm` and
`right_arm`, each containing seven revolute joints followed by one gripper
coordinate. Each rollout must contain `meta.json`, `events.jsonl`, `result.json`
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

## Pi05 pack-plate images and native predictions

The optional collector uses the existing XPolicyLab Pi05 model and artifact
validator. It does not implement another model loader or sampler. Use the Pi05
environment with `av`, `pandas`, `pyarrow`, `numpy`, `Pillow` and `PyYAML` available:

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/validation/pi05_pack_plate_chunk_speed.py \
  --dataset /path/to/pack-plate-lerobot-v3 \
  --checkpoint /path/to/export/checkpoint-59999 \
  --recipe manimux/configs/policy/pi05/tianji/pack_plate/full-zero-state-pack-instruction-step59999.yaml \
  --episodes 5,25,50,75,100,110 \
  --stride 15 --seed 20261009 \
  --prompt 'place each plate into a bubble wrap sleeve' \
  --out data/analysis/chunk-speed-offline

.venv/bin/python scripts/validation/plot_chunk_speed.py \
  --session data/experiments/EXPERIMENT/session-ID \
  --offline data/analysis/chunk-speed-offline \
  --out data/analysis/chunk-speed-comparison
```

This starts a separate GPU inference process. JAX preallocation is disabled by
default. Select available compute before running it alongside another GPU job.
The collector requires an empty output directory outside the dataset/checkpoint
and records completion only after all selected episodes finish.

Supported dataset layout: LeRobot v3 parquet/video metadata, 30 Hz, paired
`observation.images.left_wrist`/`right_wrist` videos, and 20-dimensional
`observation.state`/`action`. Each arm has XYZ metres, rotation6d and opening;
dataset order is left then right. The probe checks `action[t] == state[t+1]`
and converts the observation to the existing model's right-then-left convention.
The selected recipe must pass XPolicyLab's pack-plate zero-pose artifact contract.

Sampling uses the original RGB frames aligned by video PTS. The report embeds
smaller JPEG previews; source shapes, PTS and RGB SHA-256 remain in the evidence.
The RNG seed is `seed + episode * 100000 + frame`; each observation uses an
independent default sample without RTC conditioning. The probe writes:

- `episodes.json`: frame records, paired image previews, native prediction and
  future-GT speeds, full demonstration speed timelines and per-call latency.
- `native-actions.npz`: the exact converted native action arrays and matching
  `[episode, frame, seed]` indices, before IK, handoff or executor processing.
- `provenance.json`: checkpoint/normalization identity, dataset split, source
  hashes, sampler settings, completion status and steady-state latency summary.

The report rejects unfinished results, checkpoint or normalization mismatches,
incorrect chunk counts and invalid speed series. Dataset split and inference
conditions are shown in the report: training demonstrations are not an independent
test set. Identical checkpoint identity does not make offline images equivalent
to online observations or RTC-conditioned inference.

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
