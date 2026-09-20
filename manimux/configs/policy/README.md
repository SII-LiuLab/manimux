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
