# Inference presets

This directory supplies reusable **algorithm defaults**. An experiment selects
`inference.algorithm`, optionally references a preset through `inference.config`,
and declares its comparison parameters inline. Inline values override the preset, and
the effective variant is authoritative: a preset name alone does not establish the
executed window.

| Preset | Algorithm and schedule | Key values |
| --- | --- | --- |
| [`yam_rtc.yaml`](yam_rtc.yaml) | RTC parameters only; the experiment sets `algorithm: rtc` | 12-step window, delay 4, beta 9.1, 4-row blend |
| [`yam_rtc_multi_inflight.yaml`](yam_rtc_multi_inflight.yaml) | RTC, multi-inflight at 30 Hz observations | min 12 executed, delay 4, beta 9.1 |
| [`yam_chunktrust_ahs.yaml`](yam_chunktrust_ahs.yaml) | ChunkTrust adaptive horizon, Serial | candidates 10–50 rows |
| [`yam_chunktrust_ahs_rtc.yaml`](yam_chunktrust_ahs_rtc.yaml) | ChunkTrust adaptive horizon + RTC, single-inflight | beta 10 |
| [`yam_bid_backward.yaml`](yam_bid_backward.yaml) | BID backward, Serial | 16 samples, 16-row window |
| [`yam_bid_single_inflight.yaml`](yam_bid_single_inflight.yaml) | BID backward, single-inflight refill | 16 samples, 5-row window |
| [`yam_bid_multi_inflight.yaml`](yam_bid_multi_inflight.yaml) | BID backward, multi-inflight | 16 samples, 5-row window |
| [`yam_bid_deadline.yaml`](yam_bid_deadline.yaml) | BID backward, deadline schedule | 16 samples, 5-row window |
| [`yam_bid_aac.yaml`](yam_bid_aac.yaml) | BID candidates, AAC chooses the execution length | 16 samples |
| [`yam_act_serial.yaml`](yam_act_serial.yaml) | ACT temporal ensemble, Serial | 5-row window |

The `example-*` studies mostly write their timing inline so each method file is
self-contained. For example, `example-yam-pi05/put_bottles/rtc.yaml` runs RTC with a
32-row minimum and beta 10 and does not use `yam_rtc.yaml`; the ChunkTrust variants
reference their presets. Either form is valid: reference a preset when several variants
share it, and override only what one method changes:

```yaml
inference:
  algorithm: rtc
  config: ../../../inference/yam_rtc.yaml
  rtc: {beta: 10.0}          # this variant's change
```

- These presets select scheduler/algorithm combinations; they do not establish model
  sampling support or hardware validation.
- Model-side sampling and cache options belong to the policy recipe.
- Execution smoothing and command limits belong to `executor/` and embodiment
  control profiles. `blend_policy_steps` governs runtime chunk handoff.

Current Pi05 fixed windows are 32/50. Other policies need a window within their
actual horizon and a deliberate comparison protocol; AutoHorizon remains adaptive.
Historical settings remain evidence, not launch recommendations.

For parameter meanings, supported combinations and warmup behavior, use the
[user guide](../../../guideline.html#optional-algorithm-composition).
For model capability verification, use the
[inference skill](../../../skills/manimux-inference/SKILL.md).
