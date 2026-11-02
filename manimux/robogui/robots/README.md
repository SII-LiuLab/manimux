<a id="robogui-body-configuration"></a>

# RoboGUI body configuration

The live YAM and Tianji dashboards load their `robogui.yaml` through the generic
`RobotView` and the assembled offline `RobotModel`. Add a new body with a model
reference, camera slots and display styling. Do not duplicate its FK/IK in a
per-robot Python display adapter.

The `aloha/` and `piper/` presets use the same model-loading path for offline
preview and replay. Their display placement does not define hardware calibration
or add a robot deployment recipe. See the [ALOHA](../../../skills/manimux-station-setup/references/aloha.md)
and [PiPER](../../../skills/manimux-station-setup/references/piper.md) asset guides.

See the [display protocol](../../../skills/manimux-development/references/runtime-config.md#robogui-replay-and-recording).
