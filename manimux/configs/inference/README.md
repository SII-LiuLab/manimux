# Inference presets

Experiments explicitly select `inference.algorithm`: `manimux`, `rtc`, `aac`,
`paint`, `autohorizon`, or `act_temporal_ensemble`. Sampler capability
requirements remain unchanged.

An optional `config` supplies reusable parameters. Inline fields override that
file; references are relative to the experiment and expanded once.

`inference.strategy` optionally selects a strategy implementation. UMI uses it to
assemble measured observation history around the chosen `algorithm`; the algorithm
is no longer hidden in `policy.options.history_strategy`.

`chunk_policy_steps` retains each algorithm's original meaning, such as RTC's minimum
execution prefix or PAINT's execution window. It does not change policy action
interval, model horizon, robot control rate or executor smoothing.

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
