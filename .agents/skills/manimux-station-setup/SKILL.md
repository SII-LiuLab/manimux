---
name: manimux-station-setup
description: Connect a ManiMux installation to a developer's local YAM, Tianji or newly integrated robot and cameras. Use for SDK environments, CAN or controller addresses, device serials and local configuration binding, rather than experiment scheduling.
---

# ManiMux Station Setup

Paths below are relative to the ManiMux root. Installing the project does not discover,
name or connect a robot automatically. Bind the requested body to the user's actual devices;
do not copy the current developer's addresses, serials or checkpoint paths as universal defaults.

## Hardware connection paths

| Body | How the runtime reaches hardware | Configuration and implementation |
|---|---|---|
| YAM | `YamDualArmDriver` → `YAMRobot` → `i2rt.get_yam_robot(channel=...)` → host CAN interfaces | `configs/robots/yam/common.yaml`; `src/manimux/robots/yam/driver.py` and `src/manimux/robots/yam/arm.py` |
| Tianji | `TianjiDualArmDriver` → Marvin SDK → controller IP; TacCap SDK separately opens USB grippers | `configs/robots/tianji/common.yaml`; `src/manimux/robots/tianji/driver.py`, `src/manimux/robots/tianji/sdk.py`, `src/manimux/robots/tianji/gripper.py` |

For YAM, `left_channel` and `right_channel` must name configured Linux CAN interfaces.
Determine their physical left/right mapping; names such as `can_left` are not created by
Python installation. For Tianji, use a reachable `robot_ip` and the actual
`left_gripper_sn` / `right_gripper_sn`. The Marvin bindings are vendored under the body;
`sdk_root` can select another SDK tree. TacCap still needs its native dependency installed.
Neither driver requires launching a separate original reproduction/control repository.

Consult `envs/README.md` for hardware/model environment separation. Existing `envs/*/.venv`
directories are local assets, not installed by cloning the repository. Use explicit Python
paths; these plain hardware/model venvs must not be targeted by `uv sync` or `uv run`.
Install body dependencies using its runbook; do not invent a single all-hardware extra.

## Bind cameras separately

Camera server runs where the camera devices are accessible. Use the actual device type
and serial in `configs/cameras.yaml` or the body's camera configuration; Tianji's example
is `configs/robots/tianji/cameras.yaml`. Camera serials and gripper serials are different.

Follow physical device → configured camera name → `sensors[].options.camera_names` →
`policy.options.camera_map`. A logical name or serial alone does not establish physical
left/right placement. If placement is unknown, ask or use a preview when authorized.
Changing only a manual Viewer preview does not change the model input.

## Local configuration, as implemented today

- Shared robot settings live in `configs/robots/yam/common.yaml` and
  `configs/robots/tianji/common.yaml`. They currently include both control parameters
  and machine bindings; a separate generic inference station overlay is not implemented.
- For another installation, keep private copies under the already ignored `.local/`
  and point the local experiment's `control_profile` to the appropriate copy. Update
  copied camera/server configs and checkpoint paths as needed; retain that body's
  intended collection/deployment control settings.
- `control_profile` paths resolve relative to the referring config. Profile-owned values
  cannot be overridden with conflicting values in an infra config: see `load_config`
  in `src/manimux/config.py`. Other driver paths may be relative to the working directory;
  give commands from the repository root with paths that resolve there.
- Keep collection separate per body. YAM collection selects its runtime config through
  `manimux_config` in `configs/collection/yam/station.yaml`; share the intended local
  control profile with that body's inference recipes, not with every other body.

## Before an actual connection

Read the selected config, not another recipe's behavior: YAM's
`move_to_start_on_connect` can move during Prepare, while Tianji's `execute` and
`gripper_control` default to false. Address inspection or a request for commands does not
authorize starting motion or stopping existing services. Resolve only the missing bindings
needed for the user's task; do not add an unsolicited home/reconnect workflow.

Hand off the chosen config paths, interpreter, robot bindings, camera mapping and service
hosts. For startup commands, continue with `manimux-experiment`; do not claim that offline
configuration work proves a hardware connection.
