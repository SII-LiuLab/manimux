---
name: manimux-inference
description: Select, configure or verify ManiMux inference algorithms and model-side sampling support. Use for runtime timing, algorithm combinations, selecting YAMLs and sampler/adapter audits; use experiments for study design and scoring.
---

# ManiMux Inference

Read [development](../manimux-development/SKILL.md) before implementation and the
[model matrix](references/inference-matrix.md) for the selected deployment. Trace the
experiment through `read_experiment`, its model recipe, adapter and loaded backend.
Pi05 JAX/PyTorch, DP diffusion/flow and OpenWAM variants have different capabilities.
The live HELLO sampling modes are authoritative; a method name or YAML is not proof.

## Configure a comparison

Keep checkpoint, normalization, camera mapping, action interval/horizon and executor
fixed unless the user's comparison changes them. Start from a working deployment;
use `scripts/experiments/build_inference_matrix.py` to generate and loader-validate
variants. The matrix reference has commands. Generated presets are starting choices,
not a frozen study protocol. AAC requires explicitly selected EE-increment statistics.

Read [algorithm settings](../../../guideline.html#algorithms) and
[scheduling contracts](references/inference.md) before interpreting a step count:

- `chunk_policy_steps` is the common control, but its meaning depends on the method.
  Async `progress` uses it as a source-progress threshold and retains the full horizon;
  Async `refill` uses it as a source-prefix cap. Select the generator's `--async-trigger`
  explicitly when that distinction matters. Source progress includes trimmed rows.
- RTC/PAINT request after their progress window and continue execution during inference.
  Their `algorithm` trigger does not use the ordinary refill-time threshold.
- ACT here means temporal ensembling from the ACT paper, not an ACT-trained checkpoint.
  BID is backward-only candidate selection, not the full strong/weak-policy method.
- Sampling, admission (`single_inflight`/`multi_inflight`/serial), chunk blending and
  ensemble averaging are separate choices. Multi-inflight coalesces observations;
  it does not imply concurrent GPU model calls.

For combinations, use [supported compositions](../../../guideline.html#optional-algorithm-composition).
One base algorithm can opt into `temporal_ensemble.enabled`; BID can opt into AAC's
horizon. Adding another algorithm's parameter block does not activate that algorithm.
Show explicit enable/disable overrides when relevant. Validate with `load_config`;
report effective settings and which service must reload them.

## Preserve model/action meaning

Runtime owns timing and canonical actions; adapters own physical representation and
FK/frame conversion; model transforms own normalization, masks and padding. Rebase
relative actions against this request's observation, never response-time state.
Keep gripper semantics separate. Request reconstruction must preserve sampling fields;
verify them through the policy client's wire path.

AAC candidates use independent noise on the same observation and equivalent physical
EE motion for scoring. Return full chunks plus execution metadata so fixed-horizon
adapters decode before runtime cropping. Do not score relative rows as absolute joints.
DP's observation history and `n_obs_steps - 1` output offset remain model contracts.

## Validate and report

For implementation changes, test fixed-noise no-condition/zero-weight equivalence,
condition shapes, action/frame conversion, metadata, reset and unsupported-mode errors
as applicable. Read model-specific and nested repository instructions before editing.
Keep the pinned AutoHorizon selector separate from proposed corrections; label local
DP/OpenWAM adaptations explicitly. [Reproduction records](references/reproductions/README.md)
separate official sources, ports and measured evidence.

Distinguish loader/CPU checks, real-weight inference and physical trials. Real-model
validation checks actual specialized requests, full finite actions, candidate diversity
or guidance effects, latency/memory and reset. Offline integration does not authorize
starting services, GPU jobs or robot motion. Return to
[experiments](../manimux-experiments/SKILL.md) for collection and evaluation.
