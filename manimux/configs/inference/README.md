# Inference presets

This directory supplies reusable **algorithm defaults**. An experiment selects
`inference.algorithm`, optionally references a preset through `inference.config`,
and declares its comparison parameters. The effective variant is authoritative;
a preset name alone does not establish the executed window.

- `aligned/`: shared settings used by the current published study.
- Other YAMLs: selecting examples for scheduler/algorithm combinations; they do
  not establish model sampling support or hardware validation.
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
