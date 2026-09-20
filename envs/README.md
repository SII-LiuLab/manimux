# Local Python environments

`envs/` is a conventional location for local virtual environments, such as
`envs/yam/.venv`. It does not contain robot IPs, CAN bindings or experiment YAML.
Cloning the repository does not create these environments; a checkout containing only
this README does not imply that environments in another checkout are missing.

| Location | Purpose |
| --- | --- |
| Root `.venv/` | ManiMux core, development tools and offline Viewer; managed by the root project |
| `envs/yam/.venv/` | YAM runtime, i2rt, cameras, Viewer and optional collection dependencies |
| `envs/tianji/.venv/` | Tianji/TacCap hardware dependencies, prepared with the body runbook |
| Model-specific environment | XPolicyLab model inference/training, such as `XPolicyLab/policy/Pi_05/openpi/.venv/` |
| Existing `envs/umi_dp/`, `envs/xr1/`, etc. | Local environment conventions still referenced by some model launchers; follow their runbooks |

The hardware process does not need torch/JAX to consume Pi05 predictions from a separate
model service. An interpreter path alone does not identify the imported ManiMux checkout:
that depends on its installation and import path. Install the intended checkout into the
appropriate environment instead of treating an old interpreter path as a source migration.

## Installation entry points

- First connection to your own robot: [local station setup](../manimux/configs/local/README.md).
- YAM SDK: [YAM component README](../manimux/embodiments/arm/yam/README.md).
- Cameras: [RealSense README](../manimux/embodiments/sensor/realsense/README.md).
- Tianji/TacCap: [deployment runbook](../docs/umi-dp-tianji-taccap-runbook.md).
- Models: [deployment runbook index](../docs/README.md#policies-and-deployment).

Hardware/model environments are usually created with `uv venv`, then populated with
`uv pip install --python`. When adding dependencies, target the interpreter explicitly:

```bash
uv pip install --python envs/yam/.venv/bin/python -e '.[collection,realsense,xpolicylab]'
```

Do not point root-project `uv sync` or `UV_PROJECT_ENVIRONMENT` at an existing independent
hardware/model environment: syncing the root lockfile can remove SDK/model dependencies
not declared there. Manage the root development environment through the root project normally.

## Different from configuration

- `manimux/configs/`: experiments, assemblies, inference, executors, model-service settings
  and station templates. The private `manimux/configs/local/station.yaml` binds local devices.
- [`env_cfg/`](../env_cfg/README.md): action-field dimensions and environment batch metadata
  consumed by XPolicyLab. It is not a Python environment or a station file.

Environment locations and all model launchers have not been unified into one layout.
They should not be confused with the robot's CAN, serial and IP bindings.
