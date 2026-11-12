# Policy deployment recipes

These recipes select the model artifacts and inference settings for a ManiMux
experiment. XPolicyLab consumes their model parameters and owns model loading,
preprocessing, normalization, sampling and the shared policy server.

```text
policy/
├── pi05/yam/
│   ├── base.yaml
│   └── put-bottles/joint-step30000.yaml
├── umi_dp/tianji/
│   ├── assembly.yaml
│   └── pass_ball/default.yaml
└── <model>/<embodiment>/<recipe>.yaml
```

An experiment references one recipe through `policy_server.config`; inline
experiment values override that recipe. Several inference algorithms can reuse
the same checkpoint recipe. This directory has no `server/` or `infra/` layer:
complete robot runtime configurations belong in `configs/experiments/`.

Keep each responsibility in its owning layer:

- XPolicyLab: model implementation, model defaults and provider-specific loading.
- Policy recipe: selected checkpoint, normalization artifacts and inference choices.
- Experiment: robot assembly, observation/action adapter, scheduling and execution.
- Local station: device connections, service addresses and local storage roots.

Training-framework compatibility belongs at the XPolicyLab model/provider boundary.
A framework's exported checkpoint must retain the preprocessing, normalization and
action contract needed by its inference implementation. ManiMux's runtime continues
to consume the shared policy interface; adding a framework does not require a new
hardware control loop or a framework-specific experiment directory.

Private training recipes and launchers live under the repository root's ignored
`training/` directory. Checkpoint metadata needed to reconstruct a model at inference
time remains part of its deployment artifacts, even when named `training_config_path`.

## Model layout passed to XPolicyLab

The policy recipe owns the model-side layout, for example:

```yaml
env_cfg_type: yam_dual
robot_action_dim_info: {arm_dim: [6, 6], ee_dim: [1, 1]}
num_envs: 1
```

`env_cfg_type` remains checkpoint/profile identity. Explicit dimensions replace the
former root `env_cfg/` lookup. The experiment's `policy_server.config` loads this
recipe; inline `policy_server` fields override it through the existing merge.
`--experiment` launchers pass that resolved mapping to the model server. Standalone
`--config` launchers read the same complete recipe, without requiring a robot config.

The embodiment still owns real joint/tool layout, geometry and SDK behavior. The
recipe owns what the model consumes/emits; its adapter connects that representation
to the embodiment. An EE model's pose width is not its robot's joint count. Do not
infer or overwrite one from the other. This migration changes configuration ownership,
not packing order, action semantics, FK/IK, timing or execution behavior.

XPolicyLab's shared LeRobot converters accept `--model-config <recipe> --fps <source-rate>`.
Pi05/DP/LingBot conversion scripts also accept the recipe via `--model-config` or
`XPOLICYLAB_MODEL_CONFIG` in their shell wrappers. See the
[framework guide](../../../XPolicyLab/README.md#explicit-deployment-layout).
