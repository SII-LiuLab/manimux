# Complete experiment example

[`yam_pi05_rtc.yaml`](yam_pi05_rtc.yaml) is an annotated YAM / Pi05 joint-action /
put-bottles / step-30000 example. It composes existing robot, camera, policy-server,
inference and executor recipes. For the regular executable experiment, see
[`experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml`](../experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml).

The example sets `robot.options.execute`, `move_to_start_on_connect` and `home_on_close`
to false. It still connects to the real robot, camera and policy services when launched;
it is not an offline simulation. Choose execution/start/home settings explicitly before
using it for robot motion. Existing experiment settings have not been changed.

## Configuration ownership

- `robot`: assembled hardware and command frequency; `control_profile`: layout and limits.
- `camera_server`: devices opened by the camera service.
- `sensors`: camera streams consumed by the runtime.
- `policy`: runtime policy client, model-input mapping and action spacing/horizon.
- `policy_server`: checkpoint and sampler settings for the separate model process.
- `inference`: request scheduling, chunk handoff and timeline start alignment.
- `executor`: command smoothing and limits.
- `run`, `viewer`, `recording`: rollout lifecycle, UI and saved evidence.

For example, `inference.action_start_mode: skip_elapsed_steps` makes the action timeline
skip elapsed action points. It does not change the observation's timestamp. The shared
RTC preset already selects this value; the example repeats it to show an inline override.
Algorithm-specific prefix handling remains in the inference strategy.

References resolve relative to each YAML. If you copy this file into
`experiments/<task>/<model>/`, change component references from `../` to `../../../`.
Output paths are relative to the launch working directory. Device bindings, service
addresses and checkpoint roots belong in the [local station file](../local/README.md).

## `run`, `serve` and experiment mode

`manimux run` constructs one runtime immediately, performs one rollout and exits.
With the Viewer enabled, the runtime starts paused for Viewer control; it does not wait
for a service-level Prepare request before constructing and starting the runtime.
Connection and configured startup motion can therefore happen before Start rollout.

`manimux serve` stays available across rollouts and requires the Viewer. Prepare creates
a fresh runtime; Start rollout begins execution; Finish completes that rollout and the
service waits for the next Prepare. Both commands use `EdgeRuntime.run()` for the actual
control loop. `serve` does not start camera or model servers for you.

`run.experiment_mode` defaults to false and selects the initial Viewer evaluation mode.
Under `serve`, Prepare normal / Prepare experiment chooses the mode for each rollout:

- Normal: no human evaluation is required before preparing the next rollout.
- Experiment: after finishing, save task result, smoothness and other evaluation fields,
  or click `Skip evaluation` to continue without a label. Either choice enables the next
  Prepare once the runtime service is ready. Skipping does not write a human-label file.

Both modes can record rollouts. This flag does not select an inference algorithm or
enable robot motion. `robot.options.execute` controls YAM command execution separately.

## Launch example

Run from the repository root after preparing the environments, checkpoint and
`manimux/configs/local/station.yaml`. Use four separate terminals and reuse matching
services that are already running. Camera, policy and runtime launchers below read the
same experiment and default station; use `--local <station.yaml>` consistently to select
another station. These are launch instructions, not offline validation commands.

Camera service:

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/examples/yam_pi05_rtc.yaml
```

Viewer (open the printed browser URL; its endpoint must match the station):

```bash
envs/yam/.venv/bin/python -m manimux.viewer.dashboard \
  --robot yam --host 127.0.0.1 --port 8086
```

Policy service:

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/examples/yam_pi05_rtc.yaml
```

Runtime service:

```bash
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/examples/yam_pi05_rtc.yaml
```

For a single rollout, replace the last command with the following; do not launch both
runtime owners at once:

```bash
envs/yam/.venv/bin/python -m manimux run \
  --config manimux/configs/examples/yam_pi05_rtc.yaml
```
