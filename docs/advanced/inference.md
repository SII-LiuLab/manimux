# Scheduling and action clocks

The model predicts a chunk. An inference strategy decides when to request another
chunk and how to hand it off. `ActionTimeline` manages the accepted trajectory;
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
asynchronous adaptation. Timeline seam blending must be zero for this strategy;
executor filtering/limits remain a separate deployment choice.

Source: [official ACT implementation at the audited revision](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L191-L259).

## Specialized methods

See [reproduction records](reproductions/README.md) for upstream versions, exact
sampler hooks, adaptations and evidence limits for AAC, PAINT and AutoHorizon.
For the client/sampler boundary, read the [XPolicyLab deployment guide](../deployment/xpolicylab.md).
When comparing tuned deployments, record their actual configuration; isolating an
algorithmic effect requires controlling the other relevant choices.
