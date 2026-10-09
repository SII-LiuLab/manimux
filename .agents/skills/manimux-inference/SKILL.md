---
name: manimux-inference
description: Integrate, configure or verify ManiMux inference algorithms across Pi05, XR1, LingBot-VLA2, DP and OpenWAM. Use for sampling capability audits, model-algorithm matrices and condition conversion; use manimux-experiments for rollout scoring and station-setup for hardware binding.
---

# ManiMux Inference

Work from the user's ManiMux checkout. Read its
`.agents/skills/manimux-development/SKILL.md` and relevant nested `AGENTS.md` before
editing. The maintained implementation map, selecting commands and validation
limits live in `docs/advanced/inference-matrix.md`; read that document rather than
copying a historical capability table into a new answer.

## Choose the actual path

Trace the selected experiment through `manimux.cli.read_experiment`, its policy
server recipe, action adapter and loaded model. Distinguish Pi05 JAX from PyTorch,
DP U-Net diffusion from flow policies, and OpenWAM architecture variants. The
loaded server's HELLO sampling modes decide availability; an inherited method or
configuration name alone is insufficient. ACT in this matrix is temporal
ensembling, not a different trained policy.

For selecting YAMLs, reuse `scripts/experiments/build_inference_matrix.py` with
an existing task and checkpoint. It validates output through the real loader.
AAC requires an explicitly chosen EE-increment statistics file. Check its data
provenance before using it for a formal comparison. Preserve checkpoint horizon,
action interval, observation history, adapter, executor and station ownership.
Do not silently exchange checkpoints to fill a matrix cell. Generated presets
are not automatically an agreed study protocol.

## Preserve action meaning

Runtime owns time alignment and supplies canonical absolute joint targets.
The embodiment adapter owns FK, reference-frame conversion and native physical
representation. Model-side transforms own training normalization, masks and
padding. Keep these stages in that order. For relative actions, rebase against
this request's observation, never the response-time state. Keep gripper semantics
separate from arm deltas.

When an adapter constructs a new request dataclass, preserve the sampling
request fields. Test the prepared request through the policy client's wire path;
a sampler unit test cannot detect a dropped field. Unified OpenWAM state and
action normalization maps are different. DP returns a window starting at
`n_obs_steps - 1` and requires measured history in the configured deployment.

AAC must sample independent noise on the same observation and score equivalent
physical EE motion across native representations. Return the complete selected
chunk plus execution-length metadata so fixed-horizon adapters can decode it
before runtime trimming. Never score relative rows as absolute joints.

## Verify and describe faithfully

Check no-condition/zero-weight equivalence with fixed noise, normalized target
shapes, finite output, anchor conversion, attention capture timing, metadata,
reset behavior and unsupported-mode failures. Then verify selecting YAMLs through
the real loader. Use local test locations according to the repository policy.

Keep provenance explicit: a local DP VP-guidance extension or OpenWAM joint-flow
extension is not an official port. Do not implement an unrelated substitute under
PAINT/AutoHorizon when a backbone lacks flow inversion or action attention.
Preserve the pinned AutoHorizon selector separately from proposed corrections.

Report source/CPU checks, real-weight inference and robot trials separately.
Model validation requires the intended checkpoint, actual sampling requests,
candidate diversity or guidance effects, latency and memory measurements. Code
integration or skill creation alone does not authorize starting robots or jobs;
honor any existing authorization and otherwise finish offline work. Commit or
push only when requested.
