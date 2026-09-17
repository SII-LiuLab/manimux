# Experiment Infrastructure

Operator-facing screenshots, button meanings and the complete state flow are in the
[Viewer visual tutorial](viewer-tutorial.html). This document keeps the experiment data contract and
fair-comparison rules.

This document defines the operational contract for repeated real-robot evaluation. Model loading,
camera services and hardware preflight remain in each model runbook; the experiment layer does not
start or modify them.

## 1. Runtime entry point

```bash
# Persistent runtime service, controlled from Viewer
manimux serve --config <experiment.yaml>
```

`serve` is the runtime CLI entry point. It keeps the selected config available while Viewer
creates isolated rollouts after an operator chooses Prepare. It does not launch a Policy Server,
camera server or Viewer. The former `run` command has been removed; use a normal rollout for
deployment and debugging.

## 2. Viewer modes

Viewer offers two Prepare buttons before each rollout:

| Mode | Intended use | Human reward |
|---|---|---|
| **Prepare normal rollout** | Deployment, debugging and demonstrations | Not required; the next rollout is not blocked |
| **Prepare experiment rollout** | Formal pilot or benchmark collection | Required after every finalized rollout |

The choice is locked after Prepare so one rollout cannot change modes midway. For an experiment
rollout, also set a readable `Layout / condition ID` such as `red-ball-left-01`.

The task text shown in Viewer is not decorative: the value present when either Prepare button is
clicked is copied into that rollout config and sent to the policy.

## 3. Operator flow

After the model server, camera server, Viewer and `manimux serve` are independently ready:

1. Confirm the task command; for an experiment rollout, fill the layout or condition ID.
2. Click `Prepare normal rollout` or `Prepare experiment rollout`.
3. Wait for `PAUSED`, inspect the physical setup, then click `Start rollout`.
4. Use `Pause / Hold` only when execution must stop without ending the rollout.
5. Click `Finish & Home` after success, failure or timeout.
6. Wait for Recorder finalization and the robot's configured shutdown/home sequence.
7. If experiment mode is ON, select `success`, `failure` or `invalid`, add the smoothness score and
   failure tags, then click `Save evaluation`.
8. Prepare the next rollout only after the service reports ready.

`Pause / Hold` holds the current commanded position; it does not return home. The advanced recovery
control can request the configured home path without ending the rollout. `Finish & Home` finalizes
the episode and then follows the runtime's configured shutdown sequence.

## 4. Evidence contract

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

- `session-manifest.json` freezes the resolved config, its SHA256, ManiMux git SHA and XPolicyLab git
  SHA.
- `meta.json` records task, layout, algorithm, experiment mode and the Policy Server fingerprint.
- `canonical_raw` is the decoded policy chunk before the inference strategy.
- `infra_output` is the chunk after the selected inference strategy.
- `committed` is the final horizon accepted by Timeline after trimming or blending.
- `ticks` stores measured state, scheduled reference, executor output and command.
- `videos/index.json` stores camera timestamps, frame counts, dropped bundles and encoder errors.
- `human-label.json` exists only when an operator saves an evaluation.
- `result.json.success` means the runtime finalized normally; it is never task success.

Video recording is best-effort and asynchronous. A full video queue drops video bundles rather than
blocking the robot control loop. Formal analysis must inspect `dropped_bundles` and `error`; a damaged
recording should be marked `invalid`, not silently counted as failure.

## 5. Fair pilot checklist

Before comparing algorithms:

- use the same checkpoint, norm stats, task text, home/start state and physical layout definition;
- warm the model server before timed rollouts;
- assign an explicit layout ID and randomize algorithm order;
- freeze each algorithm config and preserve its config hash;
- count attempts, valid rollouts, invalid rollouts and safety stops separately;
- inspect the backend fingerprint so a restarted service did not load another checkpoint;
- tune on development layouts, then stop changing parameters on test layouts;
- derive automatic metrics only after matching trajectories, videos and human labels.

### Operator-randomized layout replay

For the YAM pilot, the operator may freely place task objects inside the task's declared workspace.
That freedom is sampled once per matched block, not once per algorithm:

1. Assign a readable `layout_id` and place the objects.
2. Use the initial top-camera frame as the layout reference.
3. Run every compared algorithm once in a randomized order.
4. Restore the same layout from the reference frame before each rollout.
5. Complete the configured repeats before sampling a new layout.

If the scene cannot be restored closely enough, mark that attempt `invalid`; do not silently replace
it with an easier layout for only one algorithm. A future Viewer overlay may assist restoration, but
the fairness rule does not depend on that UI feature.

See [experiment design](experiment-design.md) for the study matrix and reporting rules.
