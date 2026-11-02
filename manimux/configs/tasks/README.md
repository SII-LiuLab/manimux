# Tasks

A task file owns what is the same for every model and method that runs the task: the
instruction sent to the model, the rollout budget, the scene layouts the operator chooses
from, the score options shown after Finish, and optionally a starting pose. Variants
inherit it through `_base_` next to their study base.

| Task | Prompt | Scoring | Used by |
| --- | --- | --- | --- |
| [`put_bottles.yaml`](put_bottles.yaml) | `put_bottles_into_the_bin` | 0–4 bottles in the bin | `example-yam-pi05`, `example-yam-dream1` |
| [`pack_plate.yaml`](pack_plate.yaml) | `plate` | success / failure | `example-tianji-pi05` |
| [`coin.yaml`](coin.yaml) | `put_the_coin_into_the_bin2` | staged | evaluation report examples |
| [`stack_blocks_medium.yaml`](stack_blocks_medium.yaml) | `stack_blocks_medium` | 0–2 blocks stacked | evaluation report examples |
| [`pick_the_moving_cube.yaml`](pick_the_moving_cube.yaml) | `pick_the_moving_cube` | staged; starts from `raised_ready` | user guide |

```yaml
run:
  task_prompt: put_bottles_into_the_bin   # exactly the label the checkpoint was trained on
  max_control_steps: 6000                 # rollout budget shared by all methods
  experiment_template:                    # layouts offered before each counted trial
    name: put_bottles_into_the_bin - four bottles, ten layouts
    layout_ids: [four_01, four_02]
    require_reference: true               # each trial needs a saved reference image
evaluation:                               # one option is chosen after Finish
  kind: stages
  task_id: put_bottles
  rubric_id: put_bottles_4_v1             # change it whenever the options change
  label: Put bottles into the bin
  options:
  - {id: count_0, label: 'Bottles in bin: 0/4', score: 0.0, success: false}
  - {id: count_4, label: 'Bottles in bin: 4/4', score: 1.0, success: true}
# Optional: a named pose from embodiment/initial_poses/ that Prepare moves to.
initial_pose: {config: ../embodiment/initial_poses/yam.yaml, id: raised_ready}
robot: {options: {move_to_start_on_connect: true}}
```

The prompt must match the training label character for character; several published
checkpoints were trained on snake_case task IDs rather than sentences. Your own task
files are published only if you add them deliberately; keep private ones in
`../local/tasks/`, which is ignored.

See [tasks and scoring](../../../guideline.html#experiments) in the user guide.
