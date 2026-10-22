# Runtime examples

These files select complete runtime configurations. Launch workflows live in the
[public guide](../../../guideline.html); installation addresses belong in local/.

YAM examples come in pairs: a direct VLA run and its RPent version, which inherits it and
adds the agent profile.

| VLA | Direct | RPent |
| --- | --- | --- |
| Dream-1 (hosted API) | [dream1-yam/serial30-example.yaml](dream1-yam/README.md) | rpent_dream1_yam.yaml |
| Pi05 via XPolicyLab | [pi05-xpolicylab-yam/serial30-example.yaml](pi05-xpolicylab-yam/README.md) | rpent_pi05_yam.yaml |

- rpent_franka.yaml (selected by agents/rpent_franka.yaml) and franka_joint_policy.yaml
  cover Franka FR3.
- rpent_demo.yaml (with policy_dispatch_demo.yaml) and the rpent_gui_*.yaml variants are
  disconnected synthetic RPent examples; configure the referenced agent provider first.

Physical examples require an installed station, a matching model service (or hosted API
key) and, for RPent, measured private motion bounds. RPent examples default to
execute=false; enable it in a private experiment that inherits the example.
