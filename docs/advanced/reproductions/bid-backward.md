# BID backward-only

For optional asynchronous execution and ACT/BID/AAC composition, see the
[current runtime contract](../inference.md#optional-algorithm-composition).
The original/default reproduction described below remains available.

This integration implements only the backward-coherence criterion from
[Bidirectional Decoding](https://bid-robot.github.io/) and follows the selection rule
in `reference_files/bid_policy.py`. It is not the full BID algorithm: there is no
forward contrast, weak policy, or second checkpoint. The fixed-horizon preset does
not use AAC entropy/horizon selection; AAC's adaptive horizon is an optional combination.

For the serial preset, sample N candidate chunks of H rows, execute K rows between
replans and select

```text
score[m] = sum(rho**t * norm(candidate[m, t] - previous[t + K], 2)
               for t in range(H - K))
chosen = argmin(score)
```

The reference is the full previously selected, accepted prediction, including its
unexecuted tail. The first prediction selects candidate 0; ties select the first
minimum. Costs use per-row Euclidean distance, not squared distance. The weights
are not normalized. The preset retains the reference defaults N=16, rho=0.9, K=5.

## ManiMux integration

`manimux/runtime/bid.py` owns selection and the reference history. The shared
`RequestScheduler` owns request admission. The serial preset submits one request
after the selected K-row prefix finishes.
Each row occupies one policy interval, including the last row. Inference-time hold
does not advance the reference by wall-clock latency: K is the number of executed
source rows, matching the supplied synchronous reference. The full selected chunk
is retained only after the timeline accepts its prefix. Rejected responses never
replace the reference. Pause, Home, a new rollout, and warmup completion reset it.

For `single_inflight`, `deadline`, and `multi_inflight`, candidate decoding finishes
before selection. The runtime passes the same handoff timestamp to selection,
prefix preparation, and timeline commit. With `drop_infer_latency`, selection
ignores expired candidate rows and compares the remaining timestamps with the full
last accepted prediction. Different row-grid phases use linear interpolation of
that prediction, matching the joint timeline; reference rows trimmed at its own
commit are excluded. With `first_step_when_ready`, candidate row zero is aligned
to the current commit time instead. Observation timestamps are never rewritten.

The async execution prefix contains up to K **future** rows after latency trimming.
The full selected prediction is retained for the next backward score. No valid
reference overlap selects candidate zero, with `overlap_steps: 0`; no extrapolation
of an expired prediction is performed. A candidate with no future rows is rejected
by the timeline. Rejected/superseded responses do not change reference history.
BID favors coherence with a prior prediction; it does not guarantee measured
tracking, collision avoidance, or enough action coverage for a chosen latency.

For fixed-horizon BID, the XPolicyLab and StarVLA clients reuse their existing `aac`
sampling hook solely to obtain N independent candidates from one loaded model,
without invoking AAC's selector. The live server must advertise that hook.
The optional BID + AAC path uses the XPolicyLab client to select an adaptive horizon
from the same batch while leaving candidate selection to BID.
XPolicyLab and StarVLA source and submodule pointers are unchanged. Feature/KV-cache
reuse and batching are properties of the loaded backend, not guaranteed by this selector.

Candidates pass through the existing action adapter before scoring. The current
supported adapter family is `JointAdapter`, including its absolute-joint subclasses
that preserve full-horizon decoding. The metric is L2 over all concatenated robot
joint and gripper targets in configured group order, after gripper conversion.
There is no extra normalization, per-arm averaging, or FK-based EE metric. This is
an explicit action-space adaptation of the reference, whose example has seven action
coordinates. Arbitrary native/delta/EEF adapters are not enabled by this change;
they need candidate-decoding validation before declaring `supports_bid_backward`.

## Configuration

```yaml
inference:
  algorithm: bid_backward
  inference_schedule: serial
  chunk_policy_steps: 5
  action_start_mode: first_step_when_ready
  blend_policy_steps: 0
  max_plan_age_s: 5.0
  bid:
    num_samples: 16
    rho: 0.9
```

`chunk_policy_steps` maps to `bid.replan_policy_steps`. Require 1 <= K < H,
N >= 2 and 0 < rho <= 1. Use inline decoding and `handoff_skip_steps: 0`.
Serial requires `first_step_when_ready` and remains the default when BID omits a
schedule. Async schedules support both action-start modes. Select a preset:

| Preset | Request admission |
| --- | --- |
| `yam_bid_backward.yaml` | Serial: finish K rows, then request |
| `yam_bid_single_inflight.yaml` | Refill threshold, with no outstanding request |
| `yam_bid_deadline.yaml` | Refill threshold, after the previous request deadline |
| `yam_bid_multi_inflight.yaml` | Continuous uploads, capped at 30 Hz |

The async default trigger is `refill`; continuous requests must be explicit.
`multi_inflight` requires an explicit positive `observation_hz` and a backend
advertising streaming support. The supported streaming path is ManiMux's
`StreamingPolicyServer` with the XPolicyLab client, using the existing independent
candidate sampler. StarVLA's current blocking client supports serial/single/deadline,
not multi-inflight. A deadline controls request admission and response validity;
it does not cancel model work already running. Tune the execution prefix, refill
threshold and timeout against measured candidate-batch latency; these presets do
not establish continuous action coverage.

ACT fusion and AAC's adaptive horizon can be combined with BID; see the
[usage guide](../inference.md#optional-algorithm-composition).
RTC, PAINT and AutoHorizon combinations remain unsupported for BID.
The preset disables additional seam blending. Changing executor smoothing or
blending also changes the actions executed and should be recorded in comparisons.

The selecting experiment keeps the existing Pi05 joint step-30000 checkpoint:

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/put_bottles/pi05/yam_pi05_bid_backward_joint_step30000.yaml

envs/yam/.venv/bin/python -m manimux.cli run \
  --config manimux/configs/experiments/put_bottles/pi05/yam_pi05_bid_backward_joint_step30000.yaml
```

The experiment selects `manimux/configs/inference/yam_bid_backward.yaml`. To use an
async variant, change its `inference.config` to, for example,
`../../../inference/yam_bid_multi_inflight.yaml`; the checkpoint and adapter stay the same.
Warmup exercises the same N-candidate request and decoding path without accepting
plans. Selection costs, candidate index, and overlap length are recorded with the
chunk; accepted-plan events also report the winning index and cost.

Offline checks establish formula, transport, lifecycle, and configuration behavior.
Real checkpoint candidate diversity, GPU memory, latency, and robot task performance
remain to be measured. In particular, N candidates can materially increase inference
cost even when observation features are shared.
