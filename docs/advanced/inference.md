# Scheduling and action clocks

The model predicts a chunk. `RequestScheduler` admits requests according to transport
capacity, request frequency and a configurable trigger. The inference strategy
retains algorithm-specific readiness, sampling and fusion. `ActionTimeline` manages the accepted trajectory;
the executor samples references at the command rate. Scheduling does not change
camera acquisition timestamps or pretend an old observation is new.

## Three different counts

- `policy.horizon_policy_steps`: the model's predicted action count.
- `inference.chunk_policy_steps`: a method-dependent execution/query window.
- Model sampler steps: iterations used to generate the prediction, configured in
  the model recipe/framework.

Twelve points spaced at `1/30 s` span an execution window of about 0.4 s.
Sampling the timeline at 100 Hz does not turn that into 0.12 s. Elapsed command
time also does not prove the robot has physically reached a target.

## Serial and asynchronous execution

```yaml
inference:
  algorithm: manimux
  inference_schedule: serial
  chunk_policy_steps: 12
  blend_policy_steps: 0
```

Serial means observe → infer → execute the selected prefix → observe again.
Inference still runs in a worker; the control loop remains responsive and holds
its last command while waiting. The accepted prefix starts at commit time, keeps
its original first row and gives the final row a full action interval. Pause
invalidates the old timeline and in-flight response; resuming uses a new observation.
Serial does not wait for measured arrival, and it does not support the asynchronous
`refill_threshold_s` setting. The current serial path requires inline decoding.

The asynchronous path can request while the old chunk is executing. Delayed
responses are aligned and trimmed by the selected strategy/timeline contract.
RTC additionally requires real sampler conditioning support from the backend;
changing a YAML algorithm name cannot add missing model-side hooks.

### Continuous observation uploads (`multi_inflight`)

```yaml
inference:
  algorithm: manimux
  inference_schedule: multi_inflight
  request_trigger: continuous
  observation_hz: 30.0
  action_start_mode: drop_infer_latency
  blend_policy_steps: 4
```

The scheduler caps requests at `observation_hz` (also limited by control ticks,
algorithm readiness and transport capacity). With `request_trigger: continuous`,
ordinary asynchronous chunking requests independently of the remaining action chunk.
`multi_inflight` permits submissions before previous responses arrive. ManiMux's
`StreamingPolicyServer` extends the unchanged XPolicyLab server to run one inference
at a time and keep one replaceable waiting observation. As soon as inference finishes,
it selects the newest waiting input and sends the completed result independently.
It waits when no fresh input exists; it does not repeatedly infer the same input.
This is a transport/scheduling mode, not concurrent GPU model execution.

Scheduling and request triggers are independent:

| Setting | Meaning |
| --- | --- |
| `inference_schedule: single_inflight` | Wait for the previous request to finish before admitting another |
| `inference_schedule: multi_inflight` | Allow overlapping requests; backend keeps the newest waiting input |
| `inference_schedule: serial` | Wait for both inference completion and execution of the selected prefix |
| `inference_schedule: deadline` | Legacy ordinary schedule: first request, then resubmit after its deadline when the trigger permits |
| `request_trigger: refill` | Ordinary async: submit only below `refill_threshold_s` |
| `request_trigger: continuous` | Ordinary async: no remaining-chunk threshold |
| `request_trigger: algorithm` | Retain specialized algorithm readiness, or serial execution completion |

For ordinary async, the default trigger remains `refill` even when switching to
`multi_inflight`; continuous prediction must be explicit. For serial and specialized
algorithms the default is `algorithm`. `observation_hz` can cap either single or
multi-inflight requests; it defaults to no extra cap for existing configurations
and must be explicitly positive for multi-inflight. It does not set camera FPS or
robot control frequency. Pause, Home and post-warmup reset clear the scheduler's
rate limit when the corresponding strategy is reset.

Existing ordinary `deadline`, `single_inflight` and serial presets retain their
cadence. The previously ignored `deadline` default resolves to `single_inflight`
for RTC/PAINT/ACT, and to `serial` for AAC/AutoHorizon, preserving their algorithm
execution rules. PAINT only supports single-inflight; AAC/AutoHorizon only support
serial. Unsupported combinations fail during configuration loading. Multi-inflight
also requires inline, streaming-safe action decoding (currently `JointAdapter`)
and a backend advertising the capability. Custom strategy plugins are not enabled
for multi-inflight or continuous triggers.

ACT and RTC can also select `inference_schedule: multi_inflight`, so that the runtime
passes `RequestState.multi_flight=True`. For RTC, it records which accepted plan each
request used. If that plan has been replaced before the result arrives, the result
is rejected as `rtc_condition_plan_replaced`. This prevents a queued request from
replacing a newer plan using an obsolete condition, but can discard computed results.
RTC's minimum execution window and ACT's query interval remain algorithm gates;
neither is overridden by the scheduler. Streaming timeouts produce a rejected
response rather than killing the worker; their transport credits remain occupied
until late replies are drained, preventing unbounded uploads to a stalled server.

Action fusion is unchanged: ACT aggregates overlapping predictions, RTC conditions
sampling, and the timeline applies the explicitly selected `blend_policy_steps`.
Changing the schedule never changes blending, horizon, executor or checkpoint.
The supplied streaming presets explicitly use zero blending; when measuring a
schedule-only comparison, keep all of those settings and the request trigger equal.
Continuous prediction can increase model utilization but is not evidence of faster
individual inference, smoother motion or higher task success.

For the Pi05 joint experiment above, select one of these inference presets:
```yaml
inference:
  config: ../../../inference/yam_multi_inflight.yaml
  # Or config: ../../../inference/yam_act_multi_inflight.yaml
  # Or: ../../../inference/yam_rtc_multi_inflight.yaml
```

## Meaning of `chunk_policy_steps`

| Strategy | Meaning |
| --- | --- |
| `manimux` | Maximum source prefix; asynchronous delay trimming can reduce usable rows |
| `rtc` | Minimum execution window for requesting again, adjusted using predicted delay; old actions continue during inference |
| `paint` | Source-action query trigger; the old chunk continues while the next request runs |
| `act_temporal_ensemble` | Query interval; overlapping predictions can still contribute to the same action time |
| AAC / AutoHorizon | No equivalent fixed window; `chunk_policy_steps` is rejected |

The loader maps this field to the corresponding strategy option. Conflicting
public and internal values are rejected. Equal numbers across methods do not
establish equal effective latency, action counts or deployment behavior.

## ACT temporal ensembling

The retained upstream rule orders predictions for the same absolute action time
from oldest to newest and weights them with:

```text
w_i = exp(-coefficient * i) / sum_j exp(-coefficient * j)
```

The upstream default coefficient is `0.01`. ManiMux exposes the query interval:

```yaml
inference:
  algorithm: act_temporal_ensemble
  blend_policy_steps: 0
  temporal_ensemble:
    coefficient: 0.01
    query_interval_policy_steps: 1
```

The interval is in model action steps, not control ticks. Setting it above one
changes the original every-step query cadence and must be reported as ManiMux's
asynchronous adaptation. The ACT presets use zero timeline seam blending to retain
the ensemble output; a positive value adds a separate handoff blend. Executor
filtering/limits remain a separate deployment choice.

Source: [official ACT implementation at the audited revision](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L191-L259).

## Specialized methods

The [model sampling matrix](inference-matrix.md) lists the five YAM policy
families, selecting-config generator, model-side adaptations and validation limits.

See [reproduction records](reproductions/README.md) for upstream versions, exact
sampler hooks, adaptations and evidence limits for AAC, PAINT and AutoHorizon.
For the client/sampler boundary, read the [XPolicyLab deployment guide](../deployment/xpolicylab.md).
When comparing tuned deployments, record their actual configuration; isolating an
algorithmic effect requires controlling the other relevant choices.
