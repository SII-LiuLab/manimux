# Experiment records and optional evaluation

For everyday operation, use the [RoboGUI research guide](research.md).
Free rollouts do not require scoring. Study rollouts support optional templates,
reference images and evaluation. This checkout keeps the team study register at
`docs/experiments.md`, outside the public documentation build. Other studies may
use their own register; everyday inference does not require one.

`manimux serve` keeps one configuration available for repeated GUI-driven Prepare
requests. Each creates a fresh rollout. `manimux run` creates one runtime immediately;
with RoboGUI enabled it starts paused, but connection/startup motion can already occur.
Neither command launches camera, model or RoboGUI services for you.

## Saved evidence

```text
data/experiments/<campaign>/<algorithm>/session-*/
├── session-manifest.json
└── rollout-001/
    ├── meta.json
    ├── events.jsonl
    ├── result.json
    ├── data.zarr/
    │   ├── ticks/
    │   └── plans/000000/
    │       ├── canonical_raw/
    │       ├── infra_output/
    │       └── committed/
    ├── videos/
    │   ├── <camera>.mp4
    │   └── index.json
    └── evaluation/
        └── human-label.json
```

- `session-manifest.json` stores the resolved configuration in `config`, the entry YAML's byte
  hash in `config_sha256`, and ManiMux/XPolicyLab git SHAs. The hash is not a digest of the resolved
  configuration, all dependencies, or dirty source.
- `meta.json` records task, layout, algorithm, experiment mode and the Policy Server fingerprint.
  New experiment rollouts also record `repeat_id` and `reference_layout` (`task`, absolute `path`,
  `sha256`). The gallery task is distinct from the policy prompt and canonical evaluation task.
  These per-attempt fields are frozen at Prepare and also published for RoboGUI reconnection.
  Ordinary rollouts have no formal layout/repeat identity. Image hashes do not preserve overwritten
  files; keep formal references unchanged. Legacy episodes without these fields remain unknown.
- `canonical_raw` is the decoded policy chunk before the inference strategy.
- `infra_output` is the chunk after the selected inference strategy.
- `committed` is the final horizon accepted by Timeline after trimming or blending.
- `ticks` stores measured state, scheduled reference, and command. The command is the executor
  output recorded after `robot.send_command()` returns; it is not a hardware acknowledgment.
  New recordings omit `optimized`, which previously duplicated `command` exactly. Existing
  recordings remain unchanged and may contain that historical field.
- `videos/index.json` stores camera timestamps, frame counts, dropped bundles and encoder errors.
- `human-label.json` exists only when an operator saves an evaluation. New labels use
  `human-label-v2` without a smoothness score. Historical v1 files remain unchanged;
  their task results remain usable and their old smoothness field is ignored.
- `result.json.success` means the runtime finalized normally; it is never task success.

Video recording is best-effort and asynchronous. A full video queue drops video bundles rather than
blocking the robot control loop. Formal analysis must inspect `dropped_bundles` and `error`. Track
task, seam, and PRM eligibility separately: unusable video does not erase a saved human task outcome.


## Interpretation

- `result.json.success` means runtime completion, not task success.
- No human-label file means unreviewed, including when the operator chose Skip.
- Decoded policy actions, scheduled references, executor commands and measured
  state are distinct stages. An executor command is not a hardware acknowledgement.
- Capture timestamps, runtime ticks and video frame rates are different clocks/rates.
- `ticks.inference_ms` repeats the latest accepted latency; averaging ticks does
  not give a per-request latency distribution.
- A partial/aborted episode is not interchangeable with a finalized trial.

Session manifests record the resolved configuration and source/deployment provenance.
Check the actual manifest and saved streams before claiming a particular run can be
reproduced; checkpoint availability and hardware behavior require separate evidence.

## Choose an evaluation for your question

For deployment comparisons, each method may use its own tuned configuration. Record
those choices and the tuning budget; conclusions apply to the resulting deployment
systems. To isolate one algorithmic change, hold the other relevant settings fixed.
The framework supports both kinds of study without imposing either one.

Choose task criteria, layouts, repetitions and exclusions before a formal study.
Preserve unsuccessful attempts, missing labels and invalid-trial reasons. Use the
[experiment skill](../../.agents/skills/manimux-experiments/SKILL.md) for requested
analysis and [PRM guide](evaluation.md) for optional model-based judging.
