# XPolicyLab environment metadata

These files describe the robot dimensions and environment batch size used by
model services and dataset conversion. For CAN interfaces, controller IPs and USB
serials, start with [local station setup](../manimux/configs/local/README.md).
Python dependency environments are documented in [envs/](../envs/README.md).

| Files | Purpose |
| --- | --- |
| `yam_dual.yml`, `tianji_dual.yml`, `tianji_umi.yml`, `droid_single.yml` | Map an `env_cfg_type` to robot and environment metadata |
| `robot/_robot_info.json` | Joint and gripper dimensions used to pack and unpack model inputs and outputs |
| `sim/*.yml` | Environment batch size; the `*_real` names do not open hardware |

For example, the Pi05 recipe selects `env_cfg_type: yam_dual`: two six-joint
arms, each with one gripper value. UMI_DP selects `tianji_umi`, which references
the `tianji_dual` robot description. The pinned XPolicyLab provider also accepts
`tianji_dual` directly and uses `droid_single` for its Cosmos3 default recipe.
Those files are not all selected by the current ManiMux experiments, but they
have provider consumers.

## Why this directory remains at the repository root

In the pinned XPolicyLab revision `2077534393039cf844a744f52eb6a44cbdd1016c`,
`utils/process_data.py` resolves `get_robot_action_dim_info()`, `get_action_dim()`
and `get_batch_size()` through `../../env_cfg`. Its LeRobot v2.1 and v3.0
conversion scripts also read the parent workspace's `env_cfg/` directly.
Moving this directory alone would break those consumers.

The remaining migration is to give the provider a shared configurable metadata
root, update its runtime and conversion consumers, and then move these files once
to `manimux/configs/policy/`. Preserve action dimensions and ordering, and avoid
maintaining a second copy or a forwarding directory. XPolicyLab's training helper
`utils/get_action_dim.sh` reads its own `utils/robot/_robot_info.json`; its matching
entries must stay consistent during that change.

Connecting another installation of the same robot does not require changing this
metadata. XPolicyLab is a separate submodule and must also be initialized, with the
chosen model's dependencies installed, before its model service can run.
