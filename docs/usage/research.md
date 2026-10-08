# Run your research in RoboGUI

Start the camera, model, RoboGUI and `manimux serve` processes using the matching
[runbook](getting-started.md). The GUI then owns daily rollout operation.
The task command is the instruction sent to the policy; the experiment name,
condition and notes describe your research and do not modify model inputs.

## Free rollouts

Enter a task. Expand **Research details** for optional experiment name, condition and notes. Click **Prepare free
rollout**, then **Start rollout**. Use **Pause / Hold** when needed and **Finish**
(the selected robot may offer Finish & Home or Finish without homing).
There is no scoring step. Recording follows the experiment's `recording.enabled`.
Prepare can connect hardware and move to the configured start pose.

## Study rollouts

Choose **Prepare study rollout** when you want layout/repeat identity or optional
post-rollout evaluation. Without a template, any text layout ID and positive repeat
number are accepted; a reference image is optional. After Finish, save an evaluation
or click **Skip evaluation**. Skipping leaves the episode unlabeled.

A template adds only the constraints you select in the experiment YAML:

```yaml
run:
  experiment_name: bottle-study
  condition: baseline
  notes: ""
  experiment_template:
    name: Three scene variants
    layout_ids: [near, far, cluttered]
    repeats: 5
    require_reference: false
```

Omit `experiment_template` for unconstrained research. Omit `layout_ids` or `repeats`
inside a template to leave that dimension unconstrained. A template constrains study
rollouts; free rollouts remain available. It does not change inference, timing,
checkpoint, robot limits or the task prompt. Templates do not automatically schedule,
randomize or count a study for you.

For the earlier ten-layout study, use string IDs `"01"` through `"10"`, `repeats: 3`
and `require_reference: true`. Existing recorded identities remain readable.
[Complete configuration example](../../manimux/configs/examples/README.md).

If attaching a reference, select it in **Reference layout · Top** and enable
**Attach selected reference image**. Its ID must match the study layout, or can
supply the ID when the optional layout field is empty. Templates can require an
image. Existing numeric filenames and named images such as `near.png` are supported.
The reference capture tool accepts a user-entered ID rather than ten fixed slots.

Prepare freezes the identity and selected image fingerprint for this rollout.
Changes for the next rollout do not rewrite previous records. Templates come from
the loaded experiment; change the YAML and restart the runtime service to change them.

## Where records go

Keep generated records under the repository's top-level `data/` directory, outside
the Python package. From the repository root, an example output is:

```text
data/yam-pi05-rtc/
└── session-.../
    ├── session-manifest.json
    └── rollout-.../
        ├── meta.json
        ├── result.json
        ├── data.zarr/
        ├── events.jsonl
        ├── videos/
        └── evaluation/       # Only when evaluation is saved
```

`run.output_dir` selects the parent of session directories. A workstation can use
`paths.output_dir` in its private station file to override it. Relative experiment
output paths resolve from the launch directory; relative station paths resolve
from the station file. The GUI displays the actual session and episode paths.
The experiment name is metadata, not a filesystem path. Records are ignored by Git.
Existing experiments retain their explicitly configured output locations.

## Review in the GUI

Open **Recorded rollouts**. The session directory follows a newly connected runtime;
you can replace it with an older session's absolute path. Click **Refresh episodes**,
select a finalized episode and inspect its identity, termination reason and label status.
This reads files on the machine running RoboGUI, not on the browser's machine.

Choose a replay source:

| Source | Meaning |
| --- | --- |
| Measured state | Robot feedback recorded on each runtime tick |
| Scheduled reference | Timeline targets before the executor |
| Executor command | Commands produced by the executor; not a hardware acknowledgement |

**Open offline replay** creates a separate paused view, normally on the next port
(e.g. 8087 beside 8086). Use Play, Frame and Speed. It follows recorded tick times;
it does not interpolate or rerun the executor. Choosing another replay replaces
that replay view. Live cameras and controls stay in the original GUI.
For a remote workstation, expose/tunnel the replay port as well and use that host
in the replay URL. Closing RoboGUI also stops its replay view.

This entry point replays trajectories from finalized `data.zarr` records, including
older compatible episodes. It does not replay their videos or raw model outputs.
For an independent NPZ trajectory use [action replay](replay.md).

## Optional evaluation

Human labels and [PRM](evaluation.md) are available when they answer
your research question. `result.json.success` describes runtime completion, not
whether the robot accomplished the task. A missing human label means unreviewed.
See [record meanings](records.md) before comparing metrics.
