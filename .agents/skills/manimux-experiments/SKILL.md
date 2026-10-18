---
name: manimux-experiments
description: Build a custom ManiMux real-robot study from a supported deployment, configure tasks and scoring, collect traceable trials, and evaluate recordings. Use for new experiment setup, comparison protocols, progress, human scores, trajectory metrics and optional PRM reports.
---

# ManiMux Experiments

Work from the user's checkout. Read the [repository guide](../manimux-development/references/agent-guide.md)
before editing. This is the entry point for building or evaluating a user's own study;
use the development skill only when a required component or behavior is missing.

## Start from the user's experiment

Inspect existing configuration and decisions before asking questions. Resolve only
missing choices: robot/cameras, model/checkpoint and action contract, task, methods,
and output location. For scored comparisons also establish layouts/reset, repetitions,
order, stopping/invalid-attempt rules and metrics. Do not inherit this team's tasks,
checkpoints, ten-layout budget or scoring weights as defaults.

Establish whether the study is single-task or multi-task. If the conversation or
existing study does not make this clear, ask the user before choosing its entry point.
An already stated all-task study remains multi-task when the user switches models;
do not silently reduce its scope to the task named in a checkpoint or example YAML.
Checkpoint training coverage and requested evaluation tasks are separate facts.
If the requested multi-task weights are unavailable, report that and clarify the
checkpoint choice while preparing the requested catalog; do not substitute a
single-task launch command as though it satisfies the study.
For multiple tasks, recommend a `--config-list` catalog with Task/Method selection
in RoboGUI, rather than separate commands for every task and method. A single task
with several methods can also use a list; a single configuration can use `--config`.
Group lists by compatible model initialization, camera, robot and action contracts;
different weights or incompatible services require separate lists and service restarts.
Keep per-task prompts, scoring templates and per-choice output directories explicit,
and validate each list with `load_experiment_list` before giving its launch command.

Free inference needs no rubric, study register or PRM. For scoring, ask whether binary
success is sufficient or partial credit is wanted. Define observable outcomes and
whether final state or an achieved milestone counts; do not invent milestone weights.
Use an existing study register when supplied. This checkout's `experiments.md` is private
and optional; preserve it rather than recreating its results from memory.

## Build a new real-robot experiment

Follow the stages that remain unfinished; do not repeat an already working deployment.

1. **Bind the station.** Use [station setup](../manimux-station-setup/SKILL.md) for
   environments, devices and service addresses. Another installation of a supported
   robot needs local bindings, not a new driver. Route unsupported hardware/models to
   [development](../manimux-development/SKILL.md).
2. **Select the policy contract.** Use [inference](../manimux-inference/SKILL.md) and
   the matching deployment reference. Confirm weights, normalization, camera mapping,
   state/action representation, action interval and horizon. Runtime, camera and model
   services must use the same experiment/station; do not substitute weights for a method.
3. **Create the experiment.** Start from a matching working deployment or the
   [annotated complete example](../../../manimux/configs/examples/README.md).
   Save under `manimux/configs/experiments/<task>/<model>/`; keep device bindings in
   the private station, checkpoint/sampler choices in `policy_server`, and timing in
   `inference`. Fix relative references after copying. Give each compared setting a
   distinct `run.output_dir`. Select execution, startup motion and Home explicitly.
4. **Add optional study rules.** Put the agreed `binary`, `count` or `stages` rubric
   in `evaluation.config`; put layouts/repeats in `run.experiment_template`.
   [Rules and examples](../../../guideline.html#experiments) also explain reference
   images and repeat counting. A stage score is one total, not a sum of milestones.
   Version changed rubrics; old episode snapshots remain authoritative.
5. **Validate and launch.** Run `load_config` with the selected station without
   constructing hardware. For several compatible settings, add a
   [config list](../../../guideline.html#experiment-lists) and validate it with
   `load_experiment_list`. Lists require shared model initialization, camera, robot
   and action contracts. Deliver exact interpreter/experiment/station paths and
   [launch commands](../../../guideline.html#launch), then operate within existing
   authorization. Configuration work alone does not authorize service startup or motion.
6. **Collect and evaluate.** In serve mode select Task/Method/Layout → Prepare Run
   or Prepare Experiment → Activate Warmup → Start → Finish → save/skip evaluation.
   Prepare may connect/move the robot as configured. Each attempt gets a new rollout;
   reset the scene per protocol. Report generation is a separate offline step below.

Selection is idle-only after recovery/evaluation. The catalog is loaded at serve
startup: disk edits require a runtime restart; model/camera changes may also require
those services to restart. Never silently restart an active experiment. Templates
track layouts/repeats; they do not automatically randomize or schedule a campaign.

## Evaluate saved trials

- [Evidence and reporting](references/evidence-and-reporting.md): locate sessions,
  qualify cohorts, preserve setting identities and update progress/result rows.
- [HTML reports](references/evaluation-reports.md): create `task.yaml`, bind episodes
  and settings, select metrics, and run `python -m manimux.evaluation --task-dir ...`.
  Includes NLDLJ/NSPARC configuration and interpretation; the report index is separate
  from the runtime scoring rubric. These trajectory metrics describe references.
- [Human and seam metrics](references/human-and-seam.md): rubric-compatible scores,
  continuous versus end-hold seams, eligibility and per-episode aggregation.
- [PRM](references/prm.md): optional judge profile, video manifest and GPU evaluation.
- [Records](references/records.md): recorded fields and evidence limitations.
- [Manipulation scope](../../../manimux/evaluation/manipulation_scope/SKILL.md): only
  when drawing demonstration/inference coverage on camera frames.

Use recorded config/rubric snapshots to identify a trial, not today's YAML or folder
name alone. Keep finalized, partial, invalid and unlabelled attempts distinguishable.
Never infer task success from normal runtime completion or delete failed trials to
improve a mean. Derive metrics outside raw episodes and save cohort/provenance plus
per-metric included counts and missing reasons. Missing results remain missing.

## Deliver

For setup: link the new experiment, station/model references, optional rubric/list,
resolved timing and output paths; give validation results and the operation sequence.
For evaluation: link the versioned report and reproducible cohort, update only the
agreed study rows, and distinguish collected, evaluated and still-missing evidence.
