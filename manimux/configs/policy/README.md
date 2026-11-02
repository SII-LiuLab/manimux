# Policy deployment recipes

These recipes select the model artifacts and inference settings for a ManiMux
experiment. XPolicyLab consumes their model parameters and owns model loading,
preprocessing, normalization, sampling and the shared policy server.

Recipes are named `<model>/<embodiment>/[<task>/]<recipe>.yaml`. Published recipes:

| Recipe | Model and checkpoint | Used by |
| --- | --- | --- |
| [`pi05/yam/put-bottles/joint.yaml`](pi05/yam/put-bottles/joint.yaml) | Pi05 finetuned on YAM put-bottles, joint actions, 50 rows | [`example-yam-pi05`](../experiments/example-yam-pi05/README.md) |
| [`pi05/yam/finetune.yaml`](pi05/yam/finetune.yaml) | Pi05 YAM finetune `robocurve/pi05-yam-molmoact2`, 16 rows | `servers.pi05` default, algorithm reproductions |
| [`pi05/droid/jointpos.yaml`](pi05/droid/jointpos.yaml) | openpi `pi05_droid_jointpos`, Franka joint positions | [`example-franka`](../experiments/example-franka/_base_.yaml) |
| [`pi05/tianji/pack_plate/wrist-only-step59999.yaml`](pi05/tianji/pack_plate/wrist-only-step59999.yaml) | Pi05 Tianji pack-plate, wrist cameras, EE poses | [`example-tianji-pi05`](../experiments/example-tianji-pi05/README.md) |
| [`pi05/tianji/pack_plate/inter-hand-relative-step{20000,59999}.yaml`](pi05/tianji/pack_plate/) | Pi05 Tianji pack-plate, inter-hand-relative state | [runbook](../../../skills/manimux-inference/references/deployment/pi05-tianji-pack-plate-inter-hand-relative.md) |
| [`openwam/yam/finetune.yaml`](openwam/yam/finetune.yaml) | OpenWAM YAM finetune template; set `ckpt_dir` | [runbook](../../../skills/manimux-inference/references/deployment/openwam-yam.md) |
| [`lingbot-vla2/yam/base.yaml`](lingbot-vla2/yam/base.yaml) | LingBot-VLA2 6B base with YAM projection stats | [runbook](../../../skills/manimux-inference/references/deployment/lingbot-vla2-yam.md) |
| [`lingbot-vla2/yam/finetune.yaml`](lingbot-vla2/yam/finetune.yaml) | LingBot-VLA2 finetune template; set the artifact paths | same runbook |
| [`xiaomi-xr1/yam/base.yaml`](xiaomi-xr1/yam/base.yaml) | Xiaomi Robotics 1 5B base with YAM projection stats | `servers.xr1` default |
| [`xiaomi-xr1/tianji/pass_ball/step50000.yaml`](xiaomi-xr1/tianji/pass_ball/step50000.yaml) | Xiaomi Robotics 1 Tianji pass-ball post-train | [runbook](../../../skills/manimux-inference/references/deployment/xiaomi-xr1-tianji-taccap.md) |
| [`abc_dit/yam/put-bottles/pretrained-200k.yaml`](abc_dit/yam/put-bottles/pretrained-200k.yaml) | ABC-DiT XL 200k pretrain, zero-shot | [runbook](../../../skills/manimux-inference/references/deployment/abc-dit-yam.md) |
| [`groot/yam/finetune.yaml`](groot/yam/finetune.yaml) | GR00T N1.7 YAM finetune | [runbook](../../../skills/manimux-inference/references/deployment/gr00t-yam.md) |
| [`umi_dp/tianji/pass_ball/default.yaml`](umi_dp/tianji/pass_ball/default.yaml) | UMI Diffusion Policy template for Tianji; set the checkpoint | `servers.umi_dp` default |
| [`isaac05/libero/base.yaml`](isaac05/libero/base.yaml) | Isaac-0.5 LIBERO, offline forward check | [runbook](../../../skills/manimux-inference/references/deployment/isaac05-offline.md) |
| [`poke/yam/api.yaml`](poke/yam/api.yaml) | Hosted Poke API; no local server | [`example-yam-poke`](../experiments/example-yam-poke/serial.yaml) |

To serve your own checkpoint, copy the closest recipe, point its artifact paths at your
export below the station's `paths.checkpoints`, rename `checkpoint_variant` (it is recorded
with every trial) and keep the training task label in `task_name`. Recipes for checkpoints
you do not publish stay private: list them in `.gitignore` next to the existing entries.

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
[framework guide](../../../third_party/XPolicyLab/README.md#explicit-deployment-layout).

## Portable artifact roots

Recipes may use `${paths.checkpoints}/...` or another named `paths` root from
`local/station.yaml`; `${repo}/...` names a checked-in resource. The collection
loader and standalone recipe reader resolve these identically. Missing station
roots fail explicitly. Paths inside `paths:` are relative to the station file.
Keep user names and machine-specific absolute directories out of shared recipes.
See the [configuration guide](../../../guideline.html#experiment-config-locations).
