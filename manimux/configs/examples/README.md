# Runtime examples

These files select complete runtime configurations. Launch workflows live in the
[public guide](../../../guideline.html); installation addresses belong in local/.

- franka_joint_policy.yaml provides ordinary ManiMux policy settings without an
  agent dependency. Checkpoint-specific examples select artifacts through
  policy_server.config.
- RPent experiments: rpent_dream1_yam.yaml (hosted Dream-1) and rpent_pi05_yam.yaml
  (local Pi05) on YAM carry their agent profile and start with `manimux up`;
  rpent_franka.yaml is selected by agents/rpent_franka.yaml. One Toolkit supplies motion
  and VLA tools. See [RPent](../../../guideline.html#agent-integrations).
- rpent_demo.yaml (with policy_dispatch_demo.yaml) and the rpent_gui_*.yaml variants are
  disconnected synthetic RPent examples; configure the referenced agent provider first.

The synthetic examples use no hardware or trained model. Physical examples require an
installed station, a matching model service (or hosted API key) and measured private
motion bounds, and default to execute=false.
