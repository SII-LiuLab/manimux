# Inference presets

Experiments explicitly select `inference.algorithm`: `manimux`, `rtc`, `aac`,
`paint`, `dvac`, `autohorizon`, or `act_temporal_ensemble`. Sampler capability
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

- `skip_elapsed_steps` drops source steps whose timestamps passed during inference.
- `first_step_when_ready` keeps the selected chunk intact and starts its first step
  when the runtime accepts the result.

The runtime always preserves the real `observation_time_ns`; this mode controls
playback without relabeling that timestamp. `blend_policy_steps` is likewise the
single configured seam-blending value for every inference strategy.
