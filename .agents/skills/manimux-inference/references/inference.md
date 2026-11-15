# Inference implementation contract

For supported schedules, parameters, constraints and YAML examples, use the
[public inference guide](../../../../guideline.html#algorithms).
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

## Sigmoid overlap handoff

Select `inference.handoff: sigmoid_smooth` alongside the existing `blend` and
`waypoint` choices. It blends time-aligned joint or TCP references before the
executor, with `sigmoid_smooth.window_policy_steps: 4` and `sigma_policy_steps: 0.5`
as initial tuning values. Both count effective playback intervals. It is independent
of Direct/Smooth execution; it is not the executor's low-pass cutoff.
Use zero `blend_policy_steps` and `handoff_skip_steps`, and Async/Serial/RTC or the
existing `RetimedRtcStrategy`, including gripper protection. Async/RTC retain
`drop_infer_latency`; Serial retains `first_step_when_ready` and uses its last issued
command as a held reference (`serial_hold`), since no moving overlap remains.
Dense chunk IK is retained; step IK mixes xyz/aperture and shortest-arc rotations
before per-tick IK. Retime replaces its cubic bridge with sigmoid, retains progress
matching, and ends fusion before gripper protection/transition ramps.
The window does not schedule requests: arrange enough overlap with the existing
request settings. Initial/missing asynchronous overlap runs unchanged; short overlap is clipped
and reported in accepted-plan `sigmoid_smooth` metadata. See the
[runtime contract](../../manimux-development/references/runtime-config.md#sigmoid-overlap-handoff)
for the exact finite-window sigmoid, compatibility and recording semantics.

Custom strategy factories/instances opt in with `supports_sigmoid_handoff = True`;
compatibility is checked by capability rather than a concrete plugin path. The tossing
variants are generated from their base experiments and `sigmoid-variants.yaml`;
`python -m scripts.experiments.build_sigmoid_variants
manimux/configs/experiments/throw_ball/pi05/sigmoid-variants.yaml --check` verifies drift.

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

### Playback clocks and prefix coverage

`policy.playback_hz` retimes action rows while native checkpoint spacing and measured
observation history stay fixed. Async progress thresholds and RTC query windows count
these retimed rows. Async refill thresholds, observation-rate caps, deadlines and
measured inference/decode durations remain wall-clock quantities.

Canonical RTC conditions resample the original decoded rows at the next observation's
timestamps, including fractional row offsets; pose rows use quaternion interpolation.
If the outgoing plan already applied a handoff skip, its source clock is advanced by
that skip so conditions describe the surviving actions at their execution times.
The incoming request's future skip never shifts its submission-time condition.
They do not substitute blended timeline rows. The runtime-only `condition_time_ns`
identifies the soft-mask origin so a history wrapper can apply the model's first-row
offset without mistakenly reanchoring weights to the current control tick. Missing
required prefixes disable conditioning instead of extrapolating targets.

`action_offset_s()` is shared by UMI decoding, history alignment and train-time RTC
forecasting. A checkpoint's default offset scales with playback; an explicit
`adapter.execution_offset_s` stays in seconds. Fixed prefix counts never scale with
frequency or observed latency. A five-row prefix leaves `5 / playback_hz` seconds
between the first fixed row and the first free row (plus the offset from observation
to the first row). Observation age, inference and decoding consume this budget.
An exhausted fixed prefix may still yield an accepted free suffix; it is not a
guarantee of a smooth physical handoff. An entirely expired horizon is rejected.

Train-time RTC acceptance events include `conditioned_prefix_steps`,
`prefix_rows_remaining_at_commit` and `prefix_exhausted_at_commit`. These distinguish
requests that used no prefix from conditioned requests whose fixed rows expired
before commit. Check these alongside measured latency and rejected-plan reasons
when increasing playback speed. IK/waypoint limits and actuator smoothing can change
the executed trajectory even when sampled prefix targets match exactly.

### Experimental RTC with adaptive execution speed

`manimux.runtime.rtc.retimed:RetimedRtcStrategy` is an opt-in strategy plugin under
`algorithm: rtc`. The Processed2 Native World H50 catalog exposes it as
**RTC Adaptive Speed / 15 Hz**, alongside Serial, Async and ordinary RTC. It uses
the same checkpoint and Pi05 inference-time guidance, not the DP trained-prefix sampler.

The model still returns H50 at its native 30 Hz contract. Playback at 15 Hz gives
the full decoded path a nominal two-times duration. `inference.rtc_retime.strength`
(default 0.5) redistributes that budget toward equal path speed. Both arms use one
phase map, based on the larger per-arm combined TCP translation/rotation increment;
`rotation_radius_m` converts angular motion into a distance metric. Individual arms
need not have equal or constant speeds. Stationary intervals, gripper-motion
intervals and the request's guidance-delay prefix retain their original allocation.
This cannot identify unrecorded contact intent or remove a model's geometric reversal.

The Throw Ball catalog also offers **RTC Gripper Protected 18 / Base 30 Hz**.
It sets `rtc_retime.outside_gripper_speed_ratio: 0.5` (default 1.0, range `(0,1]`)
and `strength: 0.4`. First allocate `dt / outside_gripper_speed_ratio` to intervals
outside gripper protection, then equalize speed within the enlarged moving-interval
budget. Either arm's aperture change at least `gripper_motion_delta` (0.005), or
`gripper_event_delta` (0.08), protects the shared interval at the base `dt`.
These are predicted adjacent-row changes, with no contact sensing.
`gripper_protect_before_steps` (nonnegative integer, default 0) extends each motion
interval backward by that many original policy intervals, clipped at the chunk start.
This method sets it to 2: both preceding intervals also retain base timing and are
excluded from speed equalization. Overlapping protections merge without recursive
expansion; no future chunk is available to protect the previous chunk retroactively.
Metadata records this setting and includes the padding in `gripper_protected_intervals`.
`gripper_transition_steps` (nonnegative integer, default 0; this method: 5) adds
that many intervals outside each edge of the padded protection. After speed
equalization, interval duration becomes `dt + w * (retimed_duration - dt)`, where
`w = smoothstep(d / (steps + 1))` and `d` is interval-index distance to the nearest
protected interval, capped at `steps + 1`. Protected intervals retain `dt`; distant
intervals retain their ordinary retiming. Overlapping ramps use the nearest protection,
and the hard RTC prefix remains unchanged. No horizon renormalization follows this
taper. For uniform motion at 30/15 Hz, each five-interval ramp lasts 250 ms before
endpoint rounding; retiming uneven motion changes this duration. Playback frequency
remains adjustable. These are discrete interval allocations, not a guarantee of
continuous TCP velocity. Chunk edges clip ramps; the existing handoff bridge still
handles joining chunks. Metadata stores `gripper_transition_steps` and per-interval
`gripper_transition_weights` (null when disabled).
Stationary intervals outside gripper protection also receive the slower base allocation
but are not speed-equalized; the boundary taper still applies there.
The hard RTC delay prefix always keeps `dt`, even without
gripper motion, to preserve its overlap clock. This exception precedes half-speed timing.
RoboGUI's existing playback control remains the base frequency: 30 Hz gives nominal
30/15 Hz, 20 Hz gives 20/10 Hz. Null uses native checkpoint timing. This changes path
timing, not the 100 Hz servo loop or model contract. Smoothing, bridging, endpoint
rounding and executor limits still affect final segment/physical speed. Metadata adds
`base_playback_hz`, `outside_gripper_base_hz`, `outside_gripper_speed_ratio` and the
original `gripper_protected_intervals`; source knots record the resulting time map.
The existing methods retain their behavior with the default speed ratio of 1.0.

At commit, progress matching searches a bounded prefix (`match_window_s`, 0.6 s
beyond elapsed playback). Outgoing reference arc length since the observation is a
progress prior, not measured travel; measured pose, issued-command pose, rotation,
direction and aperture gate each candidate. The same fractional phase applies to both
arms. No fixed skip or additional timeline blend is applied. Gripper event travel
cannot be skipped. An incompatible match is rejected, retaining the outgoing plan;
normal fresh-request/expired-plan hold behavior applies. This may cause a hold if
successive proposals cannot join. Do not interpret rejection as a model fault.

A 0.20 s position/velocity correction joins the issued command to the surviving
retimed path. Aperture blending is monotone within that bridge. Dense decoded IK
paths are resampled rather than discarded; the existing executor remains responsible
for velocity/acceleration limits. The bridge changes the path locally and cannot
promise exact half-speed measured motion. Prefix removal shortens the surviving
horizon; it is not stretched back to H50. Rounding the endpoint to the playback grid
adds at most one interval.

Conditions sample the accepted **dense execution reference**, including the bridge,
on the new observation's fixed H50 playback grid. Missing future rows have zero
weight. Query progress and latency use execution time; `chunk_policy_steps: 32`
means 32 playback intervals from the observation clock (including latency), subject
to remaining-horizon delay margin, not 32 original indices after nonlinear retiming.
Model rows, execution rows and dense servo samples remain separate quantities.

Supported: complete joint chunks, or absolute TCP chunks with step IK and
`handoff: sigmoid_smooth`; one gripper per arm, single-inflight RTC,
`drop_infer_latency`, zero `handoff_skip_steps` and `blend_policy_steps`.
Chunk IK can select `handoff: blend` (original cubic bridge) or `sigmoid_smooth`
(timeline fusion after retiming, cubic bridge disabled). Sigmoid ends before the
first upcoming gripper-protected interval or transition ramp, with an explicit
`handoff_protect_from_ns` execution timestamp. Conditions sample the accepted dense
reference in its declared joint/TCP space. Pose retiming uses SLERP and never applies
joint-position bounds to TCP coordinates; step IK and command safety apply those later.
Waypoint, ACT, streaming, source caps, other custom handoffs and independently cropped
arms remain unsupported. Configuration selects this plugin explicitly.

`plans/*/canonical_raw` retains model-adapter output; `infra_output` contains execution
rows plus `metadata.rtc_retime` (source knots, original phase at execution rows,
matched phase, search score, prefix protection and timing). Optional
`runtime_trajectory` subgroups record dense rows and their clock for either stage.
The next request's recorded `condition_source` is `retimed_runtime`. Historical files
without dense subgroups cannot reconstruct a previous dense bridge exactly.

The local offline validation uses synthetic continuous handoffs and one-step shadow
transforms of historical predictions. Shadow transforms reuse historical feedback;
they are not new model predictions, a closed-loop rollout or evidence of task success.
Restart runtime after code changes, then select the new method and Prepare. The
existing catalog launcher and policy service are reused.

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
