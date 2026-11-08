# Embodiment components

Hardware is described in three layers. Device addresses and serials never appear here; the
private station file binds them.

```text
embodiment/
├── arm/, end_effector/, sensor/   # one component each: implementation, kinematics, limits
├── robot/                         # assemblies: which components, mounts, groups, Home
│   └── *_control.yaml             # control profiles: rate contract and command bounds
├── initial_poses/                 # named starting poses offered by Prepare in RoboGUI
└── sensor/cameras/                # camera-server recipes: streams to capture, by component
```

| Robot | Assembly | Control profile (optional) | Cameras | Example |
| --- | --- | --- | --- | --- |
| Dual YAM | [`robot/yam_dual.yaml`](robot/yam_dual.yaml) | [`robot/yam_control.yaml`](robot/yam_control.yaml) | [`realsense_3_views`](sensor/cameras/realsense_3_views.yaml) | `example-yam-pi05` |
| Tianji Marvin + TacCap | [`robot/tianji_taccap.yaml`](robot/tianji_taccap.yaml), [`…_pack_plate.yaml`](robot/tianji_taccap_pack_plate.yaml) | [`robot/tianji_control.yaml`](robot/tianji_control.yaml) | [`taccap_2_views`](sensor/cameras/taccap_2_views.yaml) | `example-tianji-pi05` |
| Franka FR3 (+ Robotiq) | [`robot/franka_fr3.yaml`](robot/franka_fr3.yaml), [`robot/franka_fr3_robotiq.yaml`](robot/franka_fr3_robotiq.yaml) | — | — | `example-franka` |
| Dual PiPER | [`robot/piper_dual.yaml`](robot/piper_dual.yaml) | — | — | [CAN arms](../../../skills/manimux-station-setup/references/can-arms.md) |
| Dual ARX X5 | [`robot/arx_x5_dual.yaml`](robot/arx_x5_dual.yaml) | — | — | [CAN arms](../../../skills/manimux-station-setup/references/can-arms.md) |
| ALOHA-AgileX (offline) | [`robot/aloha_agilex.yaml`](robot/aloha_agilex.yaml) | — | — | RoboGUI `--robot aloha` |

An experiment selects the assembly with `robot.config` and, where present, the control
profile with top-level `control_profile`. IK tuning profiles live beside their arm
(`arm/yam_dls_ik.yaml`, `arm/yam_dls_gate_ik.yaml`, `arm/tianji_diff_ik.yaml`) and are
selected by the policy adapter (`policy.adapter.dls_ik` / `diff_ik`).

Starting poses: YAM keeps named poses in
[`initial_poses/yam.yaml`](initial_poses/yam.yaml); a task or variant picks one with
`initial_pose: {config, id}`. Tianji keeps its Home in the assembly (`home.joints_deg`)
because recovery and Finish & Home drive to it. The Tianji assembly variants use
`_base_` to share their components and control settings and specify their own Home.

Component code, SDK notes and calibration live with each implementation under
[`manimux/embodiments/`](../../embodiments/); station bindings are described in the
[station setup skill](../../../skills/manimux-station-setup/SKILL.md).
