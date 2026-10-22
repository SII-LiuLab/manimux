# Pi05 + YAM deployment

Use a current entry from the [experiment index](../../../../../manimux/configs/experiments/README.md)
or adapt the [complete annotated example](../../../../../manimux/configs/examples/README.md).
Retired red-object/screwdriver commands and local pilot measurements are not deployment
presets; retrieve their historical revision only when investigating those runs.

## Environments and station

Prepare the YAM environment using the [component installation](../../../../../manimux/embodiments/arm/yam/README.md).
For model dependencies, initialize XPolicyLab and follow its
[Pi05 installation](../../../../../third_party/XPolicyLab/policy/Pi_05/README.md#installation).
The model interpreter is normally `third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python`;
the hardware interpreter is normally `envs/yam/.venv/bin/python`. A fresh clone does
not create either environment or supply weights. Install the intended ManiMux checkout
into the environments that launch its modules.

Use [station setup](../../../manimux-station-setup/references/station.md) for actual
CAN interfaces, camera serials, service endpoints and `paths.checkpoints`. The recipe
selects checkpoint and normalization subpaths under that root. Keep those paired;
do not override expected backend identity to bypass a mismatch.

## Model and action contract

For the current YAM joint recipes:

- Input uses three separate RGB streams, task text and 14-D robot state: left 6 joints
  plus gripper, followed by right 6 joints plus gripper. Read the experiment's
  `policy.adapter.camera_map` for stream-to-model mapping.
- Output is absolute joint targets with checkpoint-specific horizon and action interval.
  Current local 50-step checkpoints use 30 Hz action points; other checkpoints can
  differ. `robot.control_hz` is the command frequency, not the model action frequency.
- Normalization, training config, model action format, sampling seed/steps and weights
  belong to the model recipe. Runtime scheduling, executor, start/Home motion and
  limits belong to the experiment and referenced control profile.
- Joint-only and joint+EE checkpoints are different deployments. Pi05 EEF uses its
  [own adapter/runbook](pi05-yam-eef.md). Sharing a port does not make them compatible.

Inspect current values rather than copying a historical timing/executor table.
The joint-only bottle experiment is
`manimux/configs/experiments/pi05-yam-put-bottles-joint-step30000/put_bottles/rtc_joint_step30000.yaml`;
its paired model recipe is selected by `policy_server.config`.

## Offline configuration and artifact checks

Run from the repository root, substituting your experiment and station. This example
only loads configuration; it does not construct devices or load model weights:

```bash
envs/yam/.venv/bin/python - <<'PYCODE'
from pprint import pprint
from manimux.collections import load_collection

catalog = load_collection('manimux/configs/lists/pi05-latest300ep-step30000.yaml')
config = catalog.choices['coin_rtc'].config
pprint({
    'task': config['run']['task_prompt'],
    'output_dir': config['run']['output_dir'],
    'action_dt_s': config['policy']['action_dt_s'],
    'horizon': config['policy']['horizon_policy_steps'],
    'inference': config['inference'],
    'executor': config['executor'],
    'control_hz': config['robot']['control_hz'],
    'execute': config['robot']['options']['execute'],
    'start_motion': config['robot']['options']['move_to_start_on_connect'],
    'home': config['robot']['options']['home_on_close'],
})
PYCODE
```

Validate model artifact paths and print the resolved server contract without loading
weights or listening on a port:

```bash
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --config-list manimux/configs/lists/pi05-yam-put-bottles-joint-step30000-standalone.yaml --experiment-id put_bottles_rtc_joint_step30000 \
  --check
```

Append the same `--local <station.yaml>` to supported startup/check commands when
using a non-default station. A successful path check is not a real-weight forward,
backend capability handshake, connected robot or task success.

## Run and collect

Use the [four-process launch sequence](../../../../../guideline.html#launch): camera,
RoboGUI, Pi05 server and ManiMux runtime. Reuse compatible live services instead of
starting a second owner. Camera, model and runtime use the same collection/station.
`run` does not launch the other services. Prepare can connect and move the robot
according to the experiment; it is not an offline check.

For several tasks/methods sharing one deployment, use a
[validated config list](../../../../../guideline.html#experiment-lists).
Select Task/Method/Layout → Prepare Run or Prepare Experiment → Activate Warmup →
Start rollout → Finish & Home → save/skip evaluation. A layout or scoring is not
required for Prepare Run. After Finish, wait for finalization/recovery before the
next trial or service shutdown. Avoid broad process-kill commands.

A runtime code/YAML change needs a runtime restart. Changes to model initialization,
weights or normalization require the matching model service too. The currently loaded
HELLO identity/capabilities decide whether specialized sampling is available.
See [inference settings](../inference.md) and the
[model matrix](../inference-matrix.md) for algorithm selection; do not infer support
from the menu alone.

## Evidence

Use the selected experiment's `run.output_dir`, then the saved per-rollout identities
and resolved config to find the actual run. Runtime finalization is separate from
human task success. Check video indexes for missing frames/errors before using video
as evidence. [Experiment setup and evaluation](../../../manimux-experiments/SKILL.md)
links the recording schema, cohort selection and report CLI. Keep historical trials
separate from results for new settings.
