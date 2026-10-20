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

## Choose the launch workflow

Before giving runtime commands, establish whether the user wants to **manage
experiments** or **try once**. If context does not already answer this, ask the user.
Reuse the decision for an ongoing study. Follow the
[experiment workflow](../manimux-experiments/SKILL.md#start-from-the-users-experiment):
`run --config-list` manages experiments. `run --config` tries one concrete runtime
config interactively, including an existing experiment variant; records go to a
separate trial root and no study scoring/progress is enabled. No list or duplicate
config is needed. Optional `--once` connects immediately and exits after one rollout.
Use the same input flag/file and station for camera, generic policy and runtime.
Validate with `inspect --config` or `inspect --config-list` and service `--check`.
Camera/model services remain separate from the runtime command.

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

## Withdrawn model combinations

LingBot-VLA2 RTC was withdrawn on 2026-10-18 after real-checkpoint latency and
recording audits. Do not generate RTC variants or recommend historical LingBot
RTC lists. PAINT/AutoHorizon retain their separate support boundaries. A callable
sampler is not proof of real-time feasibility; require conditioned-request latency,
actual execution/condition coverage and explicit fallback accounting before
claiming a model-algorithm combination is ready for a physical study.

## Compilation settings

Before reusing or creating a deployment recipe, verify the official launch entry,
constructor defaults, effective recipe overrides and the selected sampler's compile
support. Do not copy an old `false` flag or extrapolate one attention-hook restriction
to every algorithm. Cache, precision, attention implementation and graph compilation
are distinct settings. Record the source and reason for deviations; compilation is
not proven active merely because a flag is true. Check model identity and first-call
warmup separately from steady-state timing. Do not mix compiled and eager results
under one experiment variant. See the dated audit in [the matrix](references/inference-matrix.md#compilation-audit-2026-10-18).

## Warmup must use the selected configuration

Warmup is a dry run of the current resolved runtime configuration, not a separate
sampling recipe. Use the same scheduler, strategy request, adapter, decode context,
chunk selection/trimming, IK mode, executor and command checks. Keep model precision,
compilation/cache, task prompt, horizon, window, guidance and candidate count unchanged.
Never substitute ordinary sampling for AutoHorizon/RTC/PAINT/BID or fabricate a held
pose as a conditioning trajectory. Conditions come from an isolated preview timeline.

Use real sensor/state observations; discard computed commands at the hardware boundary.
A preview is not closed-loop robot validation. Keep preview histories and counters out
of formal trials, drain worker/decoder output and acknowledge backend reset before
Start. Verify serial/progress cadence, conditioned requests, adaptive horizons, step
and chunk IK, process decoding, error visibility and zero predicted-command dispatch
with fake devices. Report cold-call, steady model-call, decode/IK and E2E separately.

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
[experiments](../manimux-experiments/SKILL.md) for data collection and evaluation.
