# DP chunk overlap comparison

`scripts/validation/compare_dp_chunk_overlap.py` compares recorded H32 UMI-DP
ManiMux asynchronous rollouts with train-time RTC. It generates a self-contained,
offline HTML report plus machine-readable metrics, without starting hardware.

## Run

From the repository root, with NumPy, Zarr and the recorded robot's offline FK
dependencies installed:

```bash
.venv/bin/python scripts/validation/compare_dp_chunk_overlap.py \
  --baseline-root data/experiments/umi-dp-tianji-pack-plate-tracker-fixed-chunk32-diff \
  --train-root data/experiments/umi-dp-tianji-pack-plate-train-time-rtc \
  --edge-fraction 0.2 --fixed-ms 100 --stationary-speed-mm-s 0.1 \
  --out data/analysis/dp-overlap-<new-analysis-id>
```

Use a new output directory for each run. Output inside either recording root is
rejected. Source recordings and human labels remain unchanged. The CLI reads each
session's saved manifest, not the current deployment recipe, to identify the
algorithm and select its offline robot assembly. Those assembly files and their
dependencies must still be available on the analysis machine.

Every `*/rollout-*/meta.json` in the two explicit roots is inventoried. Analysis
requires finalized H32 episodes and the matching manifest algorithm (`manimux`
or `train_time_rtc`). Partial, empty, mismatched and failed-processing records
retain explicit reasons. There is no filtering on task success. An empty episode
is not a zero-error episode.

## Definitions

The numerical implementation is `manimux/evaluation/chunk_overlap.py`. It reuses
the reference type and interpolation from `manimux/evaluation/trajectory.py`.

Each adjacent accepted pair is a candidate. A canonical array's first timestamp
is `observation_time_ns + source_offset_steps * dt_ns`; the saved observation
timestamp already includes the first-action offset. A committed array starts at
`start_time_ns`. This distinction matters for waypoint handoffs. Do not compare
equal array indices or add `first_action_offset_ns` a second time.

Intersect both time domains and sample a uniform grid including the exact
endpoints, with spacing at most the incoming trajectory's `dt_ns`. Interpolate
each reference in joint space on its own clock, then apply the same arm's FK.
Coverage is mandatory: no endpoint padding, clamping or extrapolation.

For TCP positions in millimetres, let `e_i = new_i - old_i`:

- Point MSE: `mean_xyz(e_i²)`, in mm². Report the first and last points separately.
- Head/tail MSE: mean point MSE over the first/last `ceil(fraction * N)` samples;
  default fraction is 20%, and the exact sample count is saved.
- Whole MSE: `mean_i,xyz(e_i²)`, in mm².
- Distance RMSE: `sqrt(mean_i(sum_xyz(e_i²)))`, in mm, equal to `sqrt(3 * MSE)`
  **within a boundary**. Averaging RMSE over boundaries/episodes does not preserve
  this identity with an independently averaged MSE.
- Mean error: `mean_i(norm(e_i))`, in mm; also report within-overlap distance P95.
- Shape RMSE: distance RMSE after subtracting each trajectory's own starting point.
- Velocity RMSE: RMS norm of the difference of finite-difference XYZ velocities,
  in mm/s. Cosine compares the two flattened velocity sequences, not absolute
  positions. If either sequence's RMS speed is at or below the threshold, cosine
  is null and its reason is recorded. The other metrics remain valid.
- Joint MSE: mean squared joint difference in rad², excluding the last coordinate
  (the Tianji gripper). Do not mix it with XYZ MSE or interpret it as orientation.
- Committed seam: old committed reference at incoming committed start versus new
  committed first point, converted to an FK position distance. This has its own
  coverage check and denominator, independent of window length.

Four views distinguish what is being compared:

1. `canonical_full`: full intersection of decoded canonical references.
2. `canonical_future`: canonical intersection starting exactly at incoming
   committed start; earlier prefix samples are excluded.
3. `committed_future`: full committed intersection from the switch.
4. `committed_fixed`: a complete fixed-duration committed window (default 100 ms)
   from the switch. A short overlap is missing rather than shortened to fit.

## Aggregation and visual inspection

Left and right arms stay separate. The default subset requires a unique
consecutive old-to-new tick plan-ID transition, with gap no greater than
`max(30 ms, 3 * median tick interval)`. Explicit canonical holds are excluded from
this subset. All candidates remain available in the report with evidence labels.
This is a conservative diagnostic filter, not complete execution validation:
the records do not supply a complete pause/resume/hold audit.

First average boundaries within each rollout, then average rollouts equally.
The displayed aggregate P95 is the mean of within-rollout P95 values, not a pooled
percentile or confidence interval. Worst values remain cohort maxima in JSON.
Every metric has its own boundary and rollout denominator. No independence or
statistical-significance claim is made for repeated rollouts in the same session.

The report includes rollout distributions, normalized overlap error profiles,
duration/error scatter, head/tail comparison, and selectable old/new XYZ traces
or distance-error curves with the switch timestamp. Normalized profiles are for
display only; they do not change the metrics. Filters select arm, view, baseline
handoff/skip subgroup, tick evidence and train-time request prefix support.
The first and last edge windows cover different physical durations when overlap
length differs; use the fixed-duration view as a complementary comparison.

`inference_submitted.prefix_in_training_support` and `prefix_steps` are joined to
the incoming plan by request sequence. Training-supported conditioned requests,
unsupported prefix lengths, and unconditioned requests can be inspected
separately. Missing request evidence is not silently classified as supported.

## Interpretation and artifacts

`canonical_raw` is a **decoded joint reference**, potentially modified by
waypoint handoff and differential IK. It is not the native DP output. The saved
arrays cannot isolate the native model's prefix residual. Committed samples may
also omit dense runtime substeps. These are reference consistency metrics, not
measured motion, motor commands, task success or an orientation/gripper score.

Different checkpoints, handoff policies, skip lengths, latency, dates and layouts
make the available cohort an exploratory system comparison, not a controlled
causal comparison of training-time RTC against another algorithm.

The output bundle contains:

- `index.html`: standalone interactive report with embedded data, no CDN or server.
- `episode-metrics.json`: full-precision boundary metrics, curves, per-rollout
  summaries, identity, missing reasons and aggregate results.
- `boundary-metrics.csv`: one row per boundary, arm and view, without dense curves.
- `summary.json`: equal-rollout aggregate statistics for the default tick subset.
- `cohort.jsonl`: every discovered attempt and its inclusion/exclusion evidence.
- `provenance.json`: exact command, package versions and analysis-source hashes.

Metadata, manifest, resolved configuration, selected arrays and tick evidence are
hashed. FK assembly hashes do not cover referenced assets or establish historical
equivalence. HTML trace coordinates are rounded for display; JSON/CSV keep full
precision. Regression checks are in `tests/unit/test_chunk_overlap.py` (local
tests follow this checkout's ignore policy).

## Separating model targets and IK

New UMI decodes store `metadata.ee_target_trace` per arm on the recorded
`canonical_raw` chunk. `model_output` contains the original absolute TCP poses
and model apertures before gripper mapping or waypoint changes. `ik_input` contains
the mapped source-knot targets actually sent into IK, including the waypoint join
knot when applicable. Each series carries its own start time and interval; poses
are `[x, y, z, qw, qx, qy, qz]`, in metres in the arm's base frame. The trace also
stores the exact selected seed joints and seed timestamp. Decoder partitions merge
these per-arm records without changing the action path. Restart the serving runtime
to load this recording addition; existing episodes cannot acquire missing targets.

Run the diagnostic on an explicit new recording:

```bash
.venv/bin/python scripts/validation/attribute_dp_overlap.py \
  --episode <session/rollout> --fixed-ms 100 \
  --out data/analysis/<new-analysis-id>/attribution.json
```

The script requires zero blending and decomposes aligned position-difference
vectors into the model-target gap, changes introduced before IK, and the difference
of post-IK reference residuals. These vectors sum to the measured reference gap;
their norms do not sum and must not be presented as causal percentages. Target XYZ
uses linear interpolation while recorded joints are interpolated before FK, so
the residual term includes that sampling convention. Neither native targets nor
these source-knot metrics are the full dense runtime path. Partial attempts can
be inspected explicitly; only continuous observed transitions with full coverage
qualify. Missing native targets are reported as missing, never inferred from FK.

`diff_ik_lag.worst_lag_mm` is a maximum **flange** residual over the entire decoded
chunk, including dense waypoints, not a TCP residual specifically at the switch
or at +100 ms. It cannot attribute an individual overlap gap by itself.
