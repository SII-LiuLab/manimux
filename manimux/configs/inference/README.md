# Inference presets

Experiments explicitly select `inference.algorithm`: `manimux`, `rtc`, `aac`,
`paint`, `autohorizon`, `bid_backward`, or `act_temporal_ensemble`. Sampler capability
requirements remain unchanged.

An optional `config` supplies reusable parameters. Inline fields override that
file; references are relative to the experiment and expanded once.

`inference.strategy` optionally selects a strategy implementation. UMI uses it to
assemble measured observation history around the chosen `algorithm`; the algorithm
is no longer hidden in `policy.options.history_strategy`.

`chunk_policy_steps` retains each algorithm's original meaning, such as RTC's minimum
execution prefix or PAINT's execution window. It does not change policy action
interval, model horizon, robot control rate or executor smoothing.

`yam_act_serial.yaml` selects ACT with serial scheduling and a five-step execution
prefix. Here `chunk_policy_steps` is K (1..H, or null for the full prediction);
asynchronous ACT keeps its existing query-interval meaning. Serial ACT retains full
predictions and aligns their overlap at commit timestamps, including inference gaps.
It requires inline absolute-joint decoding, `first_step_when_ready`, zero blend and
zero skip; no additional `temporal_ensemble.enabled` flag is needed.

For `bid_backward`, it caps the execution prefix at K future policy rows. Serial
replanning waits for that prefix; asynchronous request timing comes from the shared
scheduler. `yam_bid_single_inflight.yaml`, `yam_bid_deadline.yaml`, and
`yam_bid_multi_inflight.yaml` select time-aligned asynchronous variants.
`yam_bid_backward.yaml` selects N=16 candidates, rho=0.9, and K=5. This is BID's
backward-coherence criterion only; see the [integration record](../../../docs/advanced/reproductions/bid-backward.md).

`action_start_mode` states how a returned action chunk starts:

- `drop_infer_latency` drops source steps whose timestamps passed between observation
  and commit, including transport, inference and decoding.
- `first_step_when_ready` keeps the selected chunk intact and starts its first step
  when the runtime accepts the result.

The runtime always preserves the real `observation_time_ns`; this mode controls
playback without relabeling that timestamp. `blend_policy_steps` is likewise the
single configured seam-blending value for every inference strategy.
An accepted, decoded result takes effect at commit time; there is no additional
switch delay. Latency estimates include the elapsed preparation and inference work,
without adding an artificial wait.

Aligned experiment presets live in `aligned/`; experiments select them through
`inference.config`. Alignment does not imply completed hardware validation.

The paired Pi05 Joint 30k bottle experiments enable `run.warmup_before_start`.
Prepare runs inference continuously while holding the measured robot state; Start
drains warmup requests and waits for a backend RESET acknowledgement before fresh
formal inference. There is no minimum count, countdown, or automatic stability gate.
The RoboGUI displays warmup chunks, predicted end-effector paths and an E2E latency
curve as display-only previews. Start and Finish clear them. Finish stops new model
requests and invalidates queued WS work; an already-running model call may finish,
but its output is discarded. Warmup resumes only after another Prepare.

In `aligned/yam_rtc.yaml`, `rtc.initial_delay_policy_steps: null` selects automatic
initialization from warmup latency. The first successful call of each sampler branch
is recorded separately and excluded from calibration. Later successful E2E samples
are rounded up to policy steps; the predictor uses the maximum of the latest ten.
Starting without calibration samples is explicitly reported and uses zero until
formal accepted plans update the estimate. Numeric initial values keep their legacy
behavior. Other model experiments have not been migrated to these aligned presets.

## Optional combinations

- `yam_aac_single_inflight.yaml` / `yam_aac_multi_inflight.yaml`: asynchronous AAC.
- `yam_autohorizon_single_inflight.yaml` / `yam_autohorizon_multi_inflight.yaml`: asynchronous AutoHorizon.
- `yam_paint_multi_inflight.yaml`: streaming PAINT with reference-plan rejection.
- `yam_bid_aac.yaml`: serial BID selection with AAC's adaptive execution length.
- `yam_act_serial.yaml`: serial ACT fusion with a configurable execution prefix.

Add `temporal_ensemble: {enabled: true}` to an experiment using RTC, PAINT, BID, AAC
or AutoHorizon to fuse aligned predictions. Keep `blend_policy_steps: 0` and no
handoff skip/waypoint. The base method still controls request timing. For asynchronous
BID+AAC, override `inference_schedule: single_inflight` (or `deadline`, or `multi_inflight` with
`observation_hz`), and `action_start_mode: drop_infer_latency`. AAC's EE statistics
must match the selected embodiment; these are selecting examples, not tuned or
hardware-validated settings. See [composition contracts](../../../docs/advanced/inference.md#optional-algorithm-composition).
