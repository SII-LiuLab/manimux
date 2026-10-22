# Runtime examples

These files select complete runtime configurations. Launch workflows live in the
[public guide](../../../guideline.html); installation addresses belong in local/.

- franka_joint_policy.yaml provides ordinary ManiMux policy settings without an
  agent dependency. Checkpoint-specific examples select artifacts through
  policy_server.config.
- RPent examples (rpent_demo.yaml, rpent_franka.yaml and rpent_yam.yaml) are selected
  by the corresponding agents/rpent*.yaml recipe. One Toolkit supplies motion and
  VLA tools. See [RPent](../../../guideline.html#agent-integrations).
- rpent_gui_demo.yaml is a disconnected RoboGUI RPent example. It uses the ordinary
  Prepare/Start/Pause/Finish controls; configure the referenced agent provider first.
- policy_dispatch_demo.yaml and RPent *_native.yaml files are legacy protocol
  regression fixtures; the native environment launcher is retired. The current
  user-facing synthetic example is agents/rpent.yaml.

The synthetic example uses no hardware or trained model. Physical examples require
an installed station, matching model service and measured private motion bounds.
