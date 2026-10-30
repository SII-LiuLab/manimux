# Inference implementation contract

For supported schedules, parameters, constraints and YAML examples, use the
[public inference guide](../../../guideline.html#algorithms).
The [model sampling matrix](inference-matrix.md) records backend capabilities;
the running server's advertised sampling modes remain authoritative.

## Ownership and entry points

| Responsibility | Implementation |
| --- | --- |
| Algorithm registry and strategy factory | `manimux/runtime/inference.py` |
| Request admission, refill, rate cap and serial completion | `manimux/runtime/scheduling.py` |
| Config normalization and compatibility validation | `manimux/runtime/__init__.py` |
| Canonical action clock, trimming and handoff | `manimux/runtime/timeline.py` |
| ACT query cadence and opt-in fusion | `manimux/runtime/temporal_ensemble.py` |
| BID backward candidate selection | `manimux/runtime/bid.py` |
| Shared candidate decoding and timestamp interpolation | `manimux/runtime/candidates.py` |
| AAC selection and adaptive execution | `manimux/runtime/aac.py` |
| AutoHorizon selected-prefix execution | `manimux/runtime/autohorizon.py` |
| RTC and PAINT request conditions and acceptance | `manimux/runtime/rtc/strategy.py`, `manimux/runtime/paint.py` |

`serial` and `async` share `DefaultChunkStrategy`. Main removed the old `manimux`
selector. Do not reintroduce a second request scheduler inside an algorithm.
Adapters preserve request sampling fields and own conversion to canonical absolute
joint targets; model frameworks own sampling, normalization and native output.

## Optional algorithm composition

The factory wraps a supported base strategy with `EnsembleStrategy`. Candidate
selection and preparation share the final timeline handoff timestamp. The inline
`decode_strategy_action` hook is also used during warmup; ordinary decoding remains
unchanged. Candidate paths require an adapter declaring `supports_candidate_decoding`
and complete, untrimmed absolute-joint rows. See the
[runtime integration contract](../../manimux-development/references/runtime-config.md).

- Fixed BID retains its full selected, unfused prediction only after acceptance.
  Serial scoring shifts it by the executed prefix, matching the supplied reference.
  Async scoring interpolates overlapping future timestamps without extrapolation or
  expired leading rows. K counts future rows; optional AAC length instead caps source rows.
- ACT retains the original exponential weighting. Composed paths interpolate at exact
  action times and discard rejected history. RTC/PAINT receive the accepted fused plan
  for subsequent conditioning. BID's own reference remains unfused.
- Serial ACT retains its explicit `act_temporal_ensemble` strategy identity. It fuses
  full predictions before cropping K rows and holds the last row for one interval.
  Its reference origin is the actual commit time, including inference gaps.
- Async AAC/AutoHorizon and BID + AAC retain the selected source prefix; latency
  consumes that prefix instead of extending its end. An entirely expired prefix is
  rejected. AAC's async backward selector uses aligned accepted candidate history;
  plain serial AAC retains its original client selection path.
- RTC + ACT and PAINT streaming/composition sample conditions on the observation's
  exact time grid. Missing required prefixes fall back to unconditioned sampling;
  replies based on replaced plans are rejected. Existing standalone paths remain intact.

Pause, Home and rollout/reset transitions clear strategy and ensemble history.
Prepare alone does not run warmup: the current GUI requires Activate Warmup.
Start drains warmup and waits for acknowledged backend reset before formal requests.
Warmup previews never seed accepted-plan history.

## Validation and provenance

Use synthetic timestamps to check fractional-grid overlap, latency trimming, fixed
versus adaptive prefixes, stale-plan rejection and lifecycle resets. Exercise the
actual wire path with fake servers and validate configurations through `load_config`.
The [reproduction records](reproductions/README.md) separate reference formulas,
local adaptations and real-model/robot evidence. Runtime compatibility and offline
checks alone do not establish hardware performance.

### Train-time RTC

`algorithm: train_time_rtc` uses a prefix-trained sampler instead of inference-time
PiGDM/soft-inpaint guidance. `TrainTimeRtcInferenceStrategy` reuses RTC scheduling
and accepted-plan feedback. The pack-plate preset selects
`train_time_rtc.delay_estimator: fixed` and `fixed_prefix_steps: 5`. Every conditioned
request fixes exactly five action rows, regardless of model/decode latency or
warmup samples; the initial-delay and E2E-window settings do not change this count.
First requests, reset and unavailable committed prefixes still use unconditioned
sampling (`prefix_steps: 0`). Submission and acceptance events record
`prefix_delay_estimator: fixed` and `configured_prefix_steps: 5`, alongside the
actual `prefix_steps` on submissions and measured latency on acceptance. A late
response can have some or all fixed rows trimmed at commit; fixing the mask does
not guarantee latency coverage. The configured count must satisfy `2*prefix <= H`
for the RTC scheduler's feasibility constraint.

For dynamic operation, `train_time_rtc.delay_estimator: decode_max` (the generic default)
uses the ordinary ManiMux process-decode forecast (`decode_forecast_size: 10`,
`decode_forecast_mode: max`, bounded below by `expected_decode_s`). Only decode
latencies enter this maximum. At submission, the prefix forecast is:

`ceil(max(0, current_observation_age + last_model_trip + decode_forecast + handoff_margin - first_action_offset) / action_dt)`.

Here `last_model_trip` runs from the previous accepted request's submission to
worker completion, including preparation, queueing, transport and inference, but
excluding IK. This is a causal estimate: ordinary waypoint handoff knows the
current model latency after receiving its response, whereas trained-prefix
sampling needs its prefix before inference starts. One long model request affects
the next prediction; it is not retained for ten predictions. The first-action
offset (or explicit `adapter.execution_offset_s`) aligns the count to source row
0, as in the old waypoint join-row analysis. Neither the prefix length nor the
predicted switch time is a guarantee that inference will finish on time.

`initial_delay_policy_steps` defaults to 5 until a model-trip sample is available;
it is not a clamp or a permanent lower bound. Reset clears model-trip history.
An aggregate warmup E2E duration cannot seed this split estimator. The explicit
`delay_estimator: end_to_end_max` retains the previous behavior, including E2E
warmup calibration and `train_time_rtc.delay_buffer_size` (default 10); that window
setting has no effect in `decode_max` mode. `chunk_policy_steps` keeps RTC's
minimum-query-window meaning and the existing feasibility check.

Submission and acceptance events record `prefix_delay_estimator`, the observation
age, model/decode forecasts, margin, action offset and `predicted_prefix_steps`.
Acceptance also records `request_to_model_ms` and `measured_prefix_steps` from the
same source-row origin. The existing `measured_delay` / `rtc_delay_ms` remain E2E
diagnostics; they do not feed the `decode_max` prefix predictor.

The UMI trained-prefix history strategy reads the committed execution reference
(including dense IK samples) at each new model target timestamp. It never
interpolates the binary mask or extrapolates missing commands. An incomplete
prefix falls back to unconditioned sampling with a recorded reason. First chunks
and reset also have no prefix. Training support such as `[0, 4, 5]` is metadata,
not a clamp on dynamic inference lengths; submission events record
`prefix_in_training_support` when the checkpoint provides this metadata.

The initial integration requires `blend_policy_steps: 0`, `handoff_skip_steps: 0`
and `action_start_mode: drop_infer_latency`. The timeline trims elapsed rows once;
there is no additional crop by prefix length. Waypoint handoff remains an explicit
execution choice. See the [pack-plate deployment example](deployment/umi-dp-tianji-taccap.md#pack-plate-train-time-rtc).
