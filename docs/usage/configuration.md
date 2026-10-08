# Configure a deployment

Start from a complete experiment YAML. It composes existing components; you do not
need to write Python to select another supported checkpoint, camera set or executor.

| What you are changing | Owner |
| --- | --- |
| Task prompt, output directory, rollout budget, optional study template | Experiment `run` |
| Robot assembly and command frequency | Experiment `robot` |
| Device addresses, serials and local checkpoint paths | Private station |
| Geometry, joint order, tool mounting and TCP | Embodiment component / assembly YAML |
| Camera resolution, FPS and optional exposure overrides | Sensor / camera-set YAML |
| Model input camera mapping and action conversion | Experiment `policy.adapter` |
| Checkpoint, model configuration and normalization | Policy recipe |
| Model action spacing and predicted horizon | Experiment `policy` |
| Request cadence and chunk handoff | `inference` |
| Command generation, smoothing and configured limits | `executor` / shared control profile |
| Scene layout and display-only initial poses | RoboGUI YAML |

See the [annotated experiment](../../manimux/configs/examples/yam_pi05_rtc.yaml)
and the [configuration field reference](../../manimux/configs/README.md).

## References and overrides

`config:` references resolve relative to the YAML containing them. Mappings merge;
lists replace rather than concatenate. The private station binds installation
values without changing policy action semantics or execution settings. Its local
paths resolve relative to the station file. Relative experiment output paths resolve
from the directory where you launch the process.

Use `--local .local/station_lab.yaml` consistently across processes that support
station loading. Otherwise startup selects the experiment's `local:` reference,
then `manimux/configs/local/station.yaml`. See the [entry-point scope table](station.md#scope-and-remaining-independent-entry-points).

## Keep the clocks separate

`policy.action_dt_s` is the spacing between model trajectory points.
`robot.control_hz` is the command-loop frequency. Camera FPS is a third setting.
A 30 Hz model trajectory can be linearly interpolated by Timeline and sampled by a
100 Hz command loop. Neither setting guarantees measured physical tracking speed.

`policy.horizon_policy_steps` is the model output length.
`inference.chunk_policy_steps` controls the selected scheduling method's execution
window or query cadence; its exact meaning is [method dependent](../advanced/inference.md).
Diffusion sampling steps are a separate model-side setting.

## Apply changes

Edit the owning YAML and restart the affected service. Editing a file does not
update a process already using its resolved configuration. RoboGUI changes only
the controls it explicitly exposes; it does not hot-reload all configuration.

For a new component or behavior, follow the [development integration map](../development/README.md).
