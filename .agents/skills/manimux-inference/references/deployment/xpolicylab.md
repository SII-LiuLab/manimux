# XPolicyLab integration guide

> 2026-10-09: The old screwdriver, red-object, pass-ball and offline experiment entries below were retired. These commands are historical; select a current entry from the [experiment index](../../../../../manimux/configs/experiments/README.md). Diagnostic tools now require an explicit experiment argument.


This guide explains ManiMux integration with XPolicyLab; model-specific deployment commands belong in their respective
runbook:

- [Pi05 + YAM](pi05-yam.md)
- [GR00T N1.7 + YAM](gr00t-yam.md)
- [Xiaomi XR-1 + YAM](xiaomi-xr1-yam.md)
- [LingBot-VLA2 + YAM](lingbot-vla2-yam.md)

Configurations follow `configs/<model>/<embodiment>/`; see naming conventions in
[`manimux/configs/README.md`](../../../../../manimux/configs/README.md). `manimux/configs/experiments/pick_red_object/xpolicylab/yam_xpolicylab_smoke.yaml` is only for
generic WebSocket bridge smoke tests, not a specific model experiment.

## Source relationship

`XPolicyLab/` is a Git submodule pointing to the `Cuzyoung/XPolicyLab` fork. ManiMux pins and records
the fork commit; the fork receives updates from official `upstream`. Models still run in isolated environments and processes;
ManiMux runtime does not import model dependencies such as torch/JAX.

`--recursive` also initializes third-party sources pinned inside XPolicyLab, such as official LingBot-VLA2 and
`NVIDIA/cosmos-framework` used by Cosmos3;
users need not clone them separately into arbitrary system directories.

```bash
git clone --recursive https://github.com/SII-LiuLab/manimux.git
cd manimux
git submodule status
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

## Data flow

```text
YAM state + three RGB views + instruction
        -> ManiMux xpolicylab_ws worker
        -> msgpack/WebSocket INFER
        -> XPolicyLab policy/<MODEL>/model.py
        -> Native model sampler
        -> Standard XPolicy action keys
        -> ManiMux model-specific adapter
        -> ActionChunk(joint_position)
        -> ManiMux shared runtime + default/RTC strategy
```

Responsibilities remain explicit:

- XPolicy model adapters own model inputs/outputs, norm stats, and native samplers.
- ManiMux policy adapters own robot groups, action semantics, and required FK/IK.
- ManiMux shared runtime owns the timeline, execution, recording, and RoboGUI; InferenceStrategy owns request timing and conditions for ordinary
  chunks or RTC.
- Cameras, CAN, and robot drivers remain outside the model repository.

## YAM embodiment declaration

The policy recipe declares the model-side layout and batch size explicitly:

```yaml
env_cfg_type: yam_dual
robot_action_dim_info: {arm_dim: [6, 6], ee_dim: [1, 1]}
num_envs: 1
```

The experiment resolves `policy_server.config` and applies inline overrides. Its
launcher passes the resulting mapping to XPolicyLab, whose model adapter reads
`get_robot_action_dim_info(model_cfg)`. There is no ManiMux-root `env_cfg/` lookup.
`env_cfg_type` remains available for checkpoint/profile identity. Standalone model
recipes follow the same contract; no hardware or URDF is loaded in the server.

The standard YAM order is:

```text
left arm 6 + left gripper 1 + right arm 6 + right gripper 1 = 14
```

`robot.group_dims`, `group_prefixes`, and `gripper_dofs` must match this order.

## ManiMux dependencies

ManiMux needs only wire-protocol dependencies:

```bash
uv pip install --python envs/yam/.venv/bin/python -e ".[xpolicylab]"
```

Each model starts its XPolicy server in its own environment. Do not install model dependencies into `envs/yam`.

## New model integration checklist

**Integrate new models and reproductions under `XPolicyLab/policy/<POLICY>/`, without adding standalone native paths.**
First read the repository [development conventions](../../../manimux-development/references/agent-guide.md#learned-models-and-backend-frameworks) and
[XPolicyLab contribution rules](../../../../../XPolicyLab/CONTRIBUTING.md). Model source, loading, preprocessing, normalization,
training adaptations, and samplers belong in XPolicyLab. ManiMux retains hardware/action adapters, configurations, lightweight launchers, and
the shared runtime. A proxy shell in XPolicyLab with the actual model still in a ManiMux native server is insufficient.

The built-in ManiMux MolmoAct2 and ABC model services, HTTP clients, experiments, and action adapters
were removed. Future MolmoAct2 deployment reuses `XPolicyLab/policy/MolmoACT2/`; any ABC reintegration
must first implement and validate an XPolicyLab adapter rather than restore deleted native or HTTP paths.

Each model needs separate server/infrastructure configurations and a runbook. Confirm:

| Item | Required source |
|---|---|
| Observation keys | Model source, processor, or training configuration |
| Camera count and order | Checkpoint, model card, or training configuration |
| State/action dimensions | Embodiment configuration and stats |
| Joint / EE / delta / absolute | Model output transform |
| Action horizon and frequency | Training configuration or checkpoint metadata |
| Norm stats | Training assets paired with the checkpoint |
| RTC support | Explicit conditioning hook in the native model sampler |

If these contracts are unknown, mark the integration blocked instead of creating a configuration that only appears runnable.

## RTC contract

RTC is not automatically provided by the WebSocket worker. Write an RTC configuration only when the model adapter implements `get_action_rtc` and
conditioning enters the native denoising/sampling process. Otherwise the server
must explicitly report RTC unsupported rather than silently fall back to ordinary inference. XPolicy declares `sampling_modes` in `HELLO_ACK`;
PolicyWorker forwards these capabilities to ManiMux, and missing RTC fails before robot connection.

## Offline self-tests

```bash
PYTHONPATH=/home/ubuntu/manimux:/home/ubuntu/manimux/XPolicyLab \
  envs/yam/.venv/bin/python -m pytest \
  XPolicyLab/tests/unit/test_ws_infer_sampling.py \
  tests/unit/test_xpolicylab_plugins.py
```

These tests do not start model services, GPUs, cameras, CAN, or hardware. Check commands for specific checkpoints belong in their
model runbooks.
