# Service entry points

`manimux up --config` orchestrates these existing entry points for local single-recipe
XPolicyLab deployments; `manimux run` remains runtime-only. See the
[managed launch guide](../../guideline.html#managed-deployment) for environment selection,
readiness evidence, process ownership, logs and graceful shutdown. No launcher presses
Prepare or Start. A blocked runtime shutdown retains its attachable operator terminal.

Choose `--config <runtime.yaml>` for unscored trials, or `--config-list <list.yaml>`
for a managed study. Camera, generic policy and runtime all accept these two inputs.
Use the same file and private station in all three commands; use the [launch guide](../../guideline.html#example-study)
for commands and environment selection.

| Module | Owns |
| --- | --- |
| `manimux.servers.camera.server` | Physical cameras and image transport |
| `manimux.servers.policy` | The selected XPolicyLab deployment in its model environment |
| `manimux run` | Experiment switching, scheduling, execution and recordings |
| `manimux.robogui.dashboard` | Operator controls and visualization |

`--experiment-id` selects a service configuration when a config list contains
incompatible entries. Without it, ambiguity is rejected. Switching runtime entries
does not reload model weights or reconnect a different camera set automatically.

Camera-only component recipes use the advanced `--camera-config` flag. Camera
`--config` always means a full runtime config. The old `--experiment` spelling is
a hidden compatibility alias; do not use it in new instructions.

Model-specific launchers remain available for artifact validation and standalone
model debugging. Their `--config` selects a **model recipe**, not an experiment or
recording root. Generic policy `--check` resolves configuration only; it does not
load weights or prove inference readiness. Camera clients never open physical cameras.

OpenWAM artifact export requires both `--bind-runtime-config <output>` and
`--runtime-template <existing-experiment>`; it has no dependency on an ignored
private example. Model networks, normalization and samplers remain in XPolicyLab.
