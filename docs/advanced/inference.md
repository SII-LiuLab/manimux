# Inference usage guide

Use this guide to configure Serial, asynchronous chunking, RTC, ACT temporal
ensembling, AAC, PAINT, AutoHorizon and backward-only BID, including their supported
combinations. Start from an experiment that already works with your robot and policy;
keep its checkpoint, cameras, action representation and executor settings.

## Configure an experiment

Set `inference.algorithm` and `inference.inference_schedule` explicitly. The examples
below replace the experiment's entire `inference` section unless described as overrides.
They assume a model horizon **H = 50**; adjust the example windows for your model's
actual H. Do not shorten `policy.horizon_policy_steps` to change how long a chunk executes.

You can instead load an [inference preset](../../manimux/configs/inference/README.md)
and override its settings:

```yaml
inference:
  config: ../../../inference/yam_act_serial.yaml
  algorithm: act_temporal_ensemble
  chunk_policy_steps: 8
```

This relative path is for an experiment under
`manimux/configs/experiments/<task>/<model>/`. `config` paths are relative to the
experiment YAML. Nested mappings merge; omitted fields remain inherited. Each
section loads one preset, without recursive `config` references.
When switching methods, remove incompatible inherited limits, triggers and blending.
Some presets contain only parameters, so select the algorithm explicitly.

The examples use `policy.action_decoding: inline` (the default). Verify that the
deployment's adapter and action format meet the compatibility requirements below
before enabling a combination. See [configuration](../usage/configuration.md) and
[deployment recipes](../usage/deployments.md) for the rest of the experiment and launch procedure.

## Compatibility

“Yes” means a supported built-in runtime combination, subject to the requirements
below; “—” means unsupported. ACT means temporal ensembling, not an ACT policy checkpoint.

| Base algorithm | `serial` | `single_inflight` | `deadline` | `multi_inflight` | Add ACT ensembling |
| --- | --- | --- | --- | --- | --- |
| Ordinary chunking: `manimux` | Yes | Yes | Yes | Yes | Select `act_temporal_ensemble` instead |
| `act_temporal_ensemble` | Yes | Yes | Resolves to `single_inflight` | Yes | Already enabled |
| `rtc` | — | Yes | Resolves to `single_inflight` | Yes | Yes |
| `paint` | — | Yes | Resolves to `single_inflight` | Yes | Yes |
| `aac` | Yes | Yes | Resolves to `serial` | Yes | Yes |
| `autohorizon` | Yes | Yes | Resolves to `serial` | Yes | Yes |
| `bid_backward` | Yes | Yes | Yes | Yes | Yes |
| `bid_backward` + AAC horizon | Yes | Yes | Yes | Yes | Yes |

The `deadline` resolutions also apply with ACT enabled. The final column adds ACT
to the base method under each of its supported schedules; it does not enable other
pairs such as RTC + PAINT.

### Policy and decoding requirements

- Ordinary chunking and standalone ACT use the policy's default sampling mode.
  RTC, PAINT, AAC and AutoHorizon require the corresponding advertised sampling mode
  (`rtc`, `paint`, `aac`, `autohorizon`). BID also requires `aac` candidate sampling.
  Adding ACT does not change the base algorithm's sampling requirement. Check the
  [model sampling matrix](inference-matrix.md) and the running server's capabilities;
  a YAML setting cannot enable a missing model hook.
- Process decoding is limited to ordinary `manimux` and RTC without ACT,
  using a compatible adapter and an asynchronous
  non-streaming schedule. It is not available for serial or multi-inflight.
- Multi-inflight requires a streaming-capable client/server and adapter. The supplied
  path uses `xpolicylab_ws`, ManiMux's streaming server, and
  `manimux.policy_adapter.joint:JointAdapter`. An upstream server without the
  `multi_inflight` capability cannot run these configurations.
- BID, serial ACT, ACT added to another algorithm, and asynchronous AAC/AutoHorizon
  require complete absolute-joint trajectories with inline decoding. The supplied
  adapter is `JointAdapter`; do not pass delta-joint or end-effector actions directly
  to it. These combinations do not support dense IK or waypoint handoffs.
- Asynchronous AAC, AAC + ACT, and BID + AAC require `policy.worker: xpolicylab_ws`.
  Every use of AAC, including BID + AAC, needs an explicit `robot.config` and
  matching dual-arm end-effector increment statistics.
- Custom `inference.strategy` plugins do not support multi-inflight, continuous
  requests, serial ACT or ACT composition. A model recipe that needs its own history
  or decoding adapter cannot gain these combinations by replacing its adapter name.

## Shared settings

### Request schedules and triggers

| `inference_schedule` | When a new request may start |
| --- | --- |
| `serial` | After the selected execution prefix finishes and the preceding response or error is processed. The robot holds its last command while waiting for inference. |
| `single_inflight` | After the preceding response or error is processed, including process decoding when enabled; the current chunk may still be executing. |
| `deadline` | First request immediately when its trigger permits; later requests only after the preceding request's `policy.timeout_s` deadline, even if its reply arrived earlier. See the algorithm-specific resolutions above. |
| `multi_inflight` | Before earlier replies arrive, subject to the observation-rate cap, trigger and algorithm readiness. |

Serial waits for scheduled execution time, not measured arrival at the target.
Reaching `policy.timeout_s` alone does not release a single-inflight request; the
client must return a response or error. It also does not cancel a remote model call.

| `request_trigger` | Applies to | Behavior |
| --- | --- | --- |
| `refill` | Asynchronous ordinary chunking, BID, AAC and AutoHorizon | Submit when remaining prepared action time is **strictly below** `refill_threshold_s`. Default for these algorithms, including multi-inflight. |
| `continuous` | The same asynchronous algorithms | Submit without checking remaining action time. Schedule and rate limits still apply. |
| `algorithm` | All serial configurations, RTC, PAINT and standalone ACT | Use the selected algorithm's execution/query timing. Default and required for these cases. |

With `multi_inflight`, the server runs one model call at a time and replaces its
waiting observation with the newest available one. It starts another call when a
fresh observation is available; this is not parallel model execution. RTC/PAINT
execution windows and ACT query intervals still limit submissions.

### Time, rate and handoff parameters

| Parameter | Default | Usage |
| --- | --- | --- |
| `policy.action_dt_s` | Required | Seconds per model action row, unless `trajectory_duration_s` overrides it. Keep the checkpoint's interval. |
| `policy.horizon_policy_steps` | Required | Full predicted horizon H, not an execution limit. |
| `inference.refill_threshold_s` | `0.4` | Seconds of remaining actions at which refill becomes eligible; `0.1` means 100 ms. Use a positive value. Zero prevents refill even with an empty timeline. |
| `inference.observation_hz` | `null` | Maximum request submission rate. A finite positive value is required for multi-inflight; other schedules may also use it. It does not change camera FPS or robot control rate. |
| `policy.timeout_s` | `1.0` | Runtime request deadline, in seconds; also sets the resubmission deadline for actual deadline scheduling. |
| `policy.options.request_timeout_s` | `policy.timeout_s` for `xpolicylab_ws` | Client transport timeout, distinct from the runtime deadline. |
| `inference.max_plan_age_s` | `1.0` | Maximum observation-to-acceptance age of a plan, in seconds. Separate from the request deadline. |
| `inference.action_start_mode` | `drop_infer_latency` | Choose the playback clock described below. |
| `inference.blend_policy_steps` | `2` | Number of model action steps used for the handoff from the measured/last command to the new trajectory. `0` disables this blend. This is separate from ACT ensembling. |
| `inference.handoff_skip_steps` | `0` | Intentional action-row skip, not a latency estimate. Leave at zero for the combinations in this guide. |

Choose a refill threshold that leaves time for preparation, transport, inference
and decoding. Outside refill mode, only the default `0.4` is tolerated and ignored;
other explicit values, including `null`, are rejected. If a preset supplies another
value, use a different preset or override it to `0.4` when changing triggers.

`first_step_when_ready` starts row zero when the result is accepted. Use it for the
serial examples below. `drop_infer_latency` keeps the observation-based action
clock and discards source rows whose times have passed before acceptance. Use it
for the asynchronous examples. Neither setting changes the recorded observation
timestamp. A late reply can have too few usable rows and be rejected.

### Meaning of `chunk_policy_steps`

Step counts are **model action rows**, not control-loop ticks. The row interval dt
is normally `policy.action_dt_s`. If `policy.trajectory_duration_s` is set, it
overrides dt with `trajectory_duration_s / (H - 1)`.

K rows span `(K - 1) * dt` between their first and last targets. Ordinary serial,
serial ACT and BID additionally hold the last row for one interval, giving K * dt
for K usable rows. AAC/AutoHorizon do not add that interval. Thus 12 rows at 30 Hz
span about 0.367 s, or 0.4 s with the last-row hold; the control-loop rate does not
change these times.

| Algorithm | Meaning of `chunk_policy_steps` | Valid setting / omitted value |
| --- | --- | --- |
| Ordinary chunking | Maximum source prefix; asynchronous latency trimming can shorten it further | Integer 2 through H; omitted/null leaves `max_chunk_policy_steps` in control, whose default is null (full horizon) |
| RTC | Minimum execution window used to time the next request, adjusted for predicted delay; execution continues during inference | Positive integer up to H; omitted uses `rtc.min_execute_policy_steps`, default roughly H/2 |
| PAINT | Source-action window that triggers the next request; execution continues during inference | K must satisfy `d <= K <= H - d`; omitted uses `paint.execution_policy_steps`, default 10 |
| ACT, asynchronous | Interval between eligible queries | Integer 1 through H−1; omitted uses `temporal_ensemble.query_interval_policy_steps`, default 1 |
| ACT, serial | Execution prefix K | Integer 1 through H; omitted/null executes H |
| BID, fixed horizon | Up to K future rows after alignment; fewer if the prediction ends earlier | Integer 1 through H−1; omitted uses `bid.replan_policy_steps`, default 5 |
| AAC, AutoHorizon, BID + AAC horizon | Chosen adaptively for each prediction | Omit this field |

Use either `chunk_policy_steps` or its method-specific counterpart. Conflicting
values are rejected; setting `chunk_policy_steps: null` does not clear an inherited
counterpart. Remove/reset that field too. Do not carry `max_chunk_policy_steps` into
a specialized method.

## Individual modes

### Serial

```yaml
inference:
  algorithm: manimux
  inference_schedule: serial
  chunk_policy_steps: 12
  action_start_mode: first_step_when_ready
  blend_policy_steps: 0
```

Executes the selected prefix, including a full interval for its last row, before
observing and requesting again. Omit `chunk_policy_steps` to execute the full chunk.
`algorithm: serial` is an alias that defaults to serial and requires that schedule.

### Asynchronous chunking

```yaml
inference:
  algorithm: manimux
  inference_schedule: single_inflight
  request_trigger: refill
  refill_threshold_s: 0.4
  action_start_mode: drop_infer_latency
  blend_policy_steps: 0
```

This uses the full available prediction and refills while actions are still executing.
Optionally add `chunk_policy_steps` to cap its source prefix. Change only
`inference_schedule` to `deadline` for deadline admission, or use the
[multi-inflight example](#multi-inflight-and-continuous-requests) below.
`algorithm: async` is also accepted, but does not select single-inflight automatically.
If neither algorithm nor schedule is specified, the defaults are `manimux` and `deadline`.

### RTC

```yaml
inference:
  algorithm: rtc
  inference_schedule: single_inflight
  chunk_policy_steps: 12
  action_start_mode: drop_infer_latency
  blend_policy_steps: 0
  rtc:
    initial_delay_policy_steps: 4
    delay_buffer_size: 10
    beta: 5.0
```

The `rtc` values above are the defaults:

- `initial_delay_policy_steps` initializes latency in model steps; require
  `2 * initial_delay_policy_steps <= H`.
- `delay_buffer_size` sets the number of recent samples whose maximum predicts delay.
- `beta` controls conditioning strength and must be finite and positive.
- `initial_delay_policy_steps: null` initializes from pre-Start warmup samples.
  To use that workflow, enable `run.warmup_before_start: true` with RoboGUI and inline
  decoding. Without calibration samples, the initial estimate is zero. See the
  [preset guide](../../manimux/configs/inference/README.md) for warmup controls.

### ACT temporal ensembling

```yaml
inference:
  algorithm: act_temporal_ensemble
  inference_schedule: single_inflight
  chunk_policy_steps: 1
  action_start_mode: drop_infer_latency
  blend_policy_steps: 0
  temporal_ensemble:
    coefficient: 0.01
```

`coefficient` must be non-negative: zero weights overlapping predictions equally;
positive values favor older contributing predictions. The default is `0.01`.
No `temporal_ensemble.enabled` flag is needed for this algorithm.

Standalone asynchronous ACT permits a positive `blend_policy_steps`, which adds a
separate handoff blend. Keep it zero to use only temporal ensembling.

#### Serial ACT with a configurable execution horizon

```yaml
inference:
  algorithm: act_temporal_ensemble
  inference_schedule: serial
  chunk_policy_steps: 5
  action_start_mode: first_step_when_ready
  blend_policy_steps: 0
  temporal_ensemble:
    coefficient: 0.01
```

Full predictions contribute at their actual execution times, including inference
gaps. Set K below H to leave potential overlap; K = H or long gaps may leave only
the current prediction. `query_interval_policy_steps` is unused in serial mode.
Keep blending and handoff skip at zero.

### AAC

```yaml
inference:
  algorithm: aac
  inference_schedule: serial
  action_start_mode: first_step_when_ready
  blend_policy_steps: 0
  aac:
    num_samples: 20
    motion_threshold: 0.2
    ee_stats_path: manimux/policies/xpolicylab/norm_stats/yam_60ep_ee_increment.json
    chunk_id_selector: "0"
    backward_beta: 0.99
```

This example uses YAM statistics and a YAM preset threshold. Replace them for a
different embodiment or dataset. Relative `ee_stats_path` values resolve from the
launch working directory, unlike `config`; run from the repository root for this
example or use an absolute path.

| AAC parameter | Usage |
| --- | --- |
| `num_samples` | Number of independent candidate chunks, at least 2; default 20. More samples increase inference/memory demand. |
| `motion_threshold` | Finite non-negative threshold used to choose the execution horizon; larger values can raise the minimum execution length. Core default is 3.0; the example uses 0.2. |
| `ee_stats_path` | Required normalization statistics for dual-arm end-effector increments. |
| `chunk_id_selector` | `"0"`: first candidate; `"mean"`: candidate closest to the mean; `"backward"`: continuity-based selection using previous predictions. Quote `"0"` in YAML. |
| `backward_beta` | Decay for `"backward"` selection, in `(0, 1]`; default 0.99. |

AAC's `"backward"` selector is distinct from `algorithm: bid_backward`.

### PAINT

```yaml
inference:
  algorithm: paint
  inference_schedule: single_inflight
  chunk_policy_steps: 12
  action_start_mode: drop_infer_latency
  blend_policy_steps: 0
  paint:
    initial_delay_policy_steps: 4
    delay_buffer_size: 10
```

Use a positive initial delay **d** and execution/query window **K** satisfying
`d <= K <= H - d`. The defaults are d = 4, K = 10 and a delay buffer of 10 samples.
Unlike RTC, PAINT does not accept `null` for its initial delay. Its delay estimate
must cover observation age, queueing, transport, inference and decoding; higher
observed delays can make the window infeasible. Select K and d for the actual model
horizon and measured latency.

### AutoHorizon

```yaml
inference:
  algorithm: autohorizon
  inference_schedule: serial
  action_start_mode: first_step_when_ready
  blend_policy_steps: 0
```

The policy chooses the execution length; omit fixed chunk limits. There is no
runtime `autohorizon` parameter block. Model-side settings belong to the
[model recipe and sampling matrix](inference-matrix.md).

### BID (backward-only)

```yaml
inference:
  algorithm: bid_backward
  inference_schedule: serial
  chunk_policy_steps: 5
  action_start_mode: first_step_when_ready
  blend_policy_steps: 0
  bid:
    num_samples: 16
    rho: 0.9
    execution_horizon: fixed
```

Uses candidates from one policy and BID's backward criterion; no forward criterion
or second policy is used.

- `num_samples`: integer at least 2, default 16.
- `rho`: backward score decay, in `(0, 1]`, default 0.9. Smaller values emphasize
  earlier overlapping action times more strongly.
- `execution_horizon`: `fixed` (default), using K, or `aac` as shown below.
- Fixed K must be smaller than H. Keep `handoff_skip_steps: 0`.

Serial BID requires `first_step_when_ready`. Fixed-horizon asynchronous BID accepts
either action-start mode; the examples use `drop_infer_latency` to discard expired rows.
Asynchronous scoring compares overlapping future action times; its K counts future
rows remaining after latency alignment, not already-expired source rows.

## Optional algorithm composition

Choose one base `algorithm`. Add ACT with `temporal_ensemble.enabled`, or select
AAC's execution horizon through `bid.execution_horizon`. Merely adding an `rtc`,
`paint` or `aac` parameter block does not activate a second algorithm.

### Add ACT to RTC, PAINT, BID, AAC or AutoHorizon

For example, RTC + ACT using a preset:

```yaml
inference:
  config: ../../../inference/yam_rtc.yaml
  algorithm: rtc
  inference_schedule: single_inflight
  blend_policy_steps: 0
  temporal_ensemble:
    enabled: true
    coefficient: 0.01
```

Apply the same `temporal_ensemble` block to PAINT, BID, AAC or AutoHorizon. Require
inline complete absolute-joint decoding, `blend_policy_steps: 0`,
`handoff_skip_steps: 0`, and the default `handoff: blend`. RTC/PAINT + ACT require
`drop_infer_latency`; serial combinations use `first_step_when_ready`.

The base algorithm retains its query/execution timing; ACT's
`query_interval_policy_steps` is unused in these combinations. Only overlapping
action times contribute, so serial AAC/AutoHorizon may have no effective overlap.

### Asynchronous adaptive horizons

To use AAC or AutoHorizon with single-inflight, keep the selected method's parameters
and replace its serial settings with these overrides:

```yaml
inference:
  inference_schedule: single_inflight
  request_trigger: refill
  refill_threshold_s: 0.4
  action_start_mode: drop_infer_latency
  blend_policy_steps: 0
```

Merge these overrides into the base example. They also work for BID and BID + AAC.
Asynchronous AAC, AutoHorizon and BID + AAC require `drop_infer_latency`, zero
blending/skip, and the decoding/client requirements above. ACT may also be added.

For an adaptive horizon of e source rows, if d rows expire before acceptance, only
the remaining e−d rows are eligible. When d reaches e, the reply is rejected. The
runtime does not extend the selected horizon to compensate for latency; short
horizons and slow inference can therefore cause holds.

### Multi-inflight and continuous requests

Ordinary asynchronous chunking with continuous observation uploads:

```yaml
inference:
  algorithm: manimux
  inference_schedule: multi_inflight
  request_trigger: continuous
  observation_hz: 30.0
  action_start_mode: drop_infer_latency
  blend_policy_steps: 0
```

For BID, AAC or AutoHorizon, keep its `algorithm` and method parameters and apply
the other settings in this example, including `action_start_mode`. For refill-based uploads,
use `request_trigger: refill` and a positive `refill_threshold_s` instead.
`continuous` can also be used with `single_inflight` or actual `deadline` scheduling.

RTC, PAINT and standalone ACT require `request_trigger: algorithm`. For example,
RTC + ACT + multi-inflight:

```yaml
inference:
  config: ../../../inference/yam_rtc_multi_inflight.yaml
  algorithm: rtc
  inference_schedule: multi_inflight
  request_trigger: algorithm
  observation_hz: 30.0
  temporal_ensemble:
    enabled: true
    coefficient: 0.01
```

For standalone RTC, PAINT or ACT, start from its own individual example and set
`inference_schedule: multi_inflight`, `observation_hz`, and `request_trigger: algorithm`.
Keep its algorithm and method parameters. Streaming PAINT requires
`drop_infer_latency` and zero blending/skip. Add ACT to RTC/PAINT as shown above.

RTC/PAINT replies can be rejected when the plan they were conditioned on has already
been replaced. PAINT also rejects replies that arrive beyond the usable conditioned
prefix. A higher upload rate can increase such rejections; it does not remove model
latency or guarantee uninterrupted execution.

### BID with AAC's execution horizon

Use the dedicated preset to avoid inheriting a fixed K:

```yaml
inference:
  config: ../../../inference/yam_bid_aac.yaml
  algorithm: bid_backward
  inference_schedule: serial
  bid:
    execution_horizon: aac
    num_samples: 16
    rho: 0.9
```

The preset supplies serial scheduling, `first_step_when_ready`, zero blending,
YAM EE statistics and AAC's motion threshold. Override `aac.ee_stats_path` and
`aac.motion_threshold` for your deployment as described in the AAC section.

`bid.num_samples` sets the shared batch size. AAC's sample-count/selector settings
and BID's fixed replan length do not control this combination; leave them at defaults.
Omit `chunk_policy_steps`, or set it to `null` to clear an inherited fixed K.
Apply the asynchronous overrides above for single-inflight or multi-inflight;
`deadline` also works. To add ACT, use the earlier `temporal_ensemble` block.

### Unsupported combinations

- RTC + PAINT, and RTC/PAINT with BID or AAC candidate selection.
- AutoHorizon with RTC, PAINT, BID or AAC sampling/horizon selection.

## Further configuration

For model-specific sampler options, use the [sampling matrix](inference-matrix.md).
For executor smoothing, limits and IK, use [execution and IK](execution.md).
Algorithm sources and derivations are kept in the [reproduction records](reproductions/README.md);
advanced decoder/waypoint configuration is covered by the
[runtime configuration reference](../development/runtime-config.md).
