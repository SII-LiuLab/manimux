<a id="robogui-body-configuration"></a>

# RoboGUI body configuration

The live YAM and Tianji dashboards load their `robogui.yaml` through the generic
`RobotView` and the assembled offline `RobotModel`. Add a new body with a model
reference, camera slots and display styling. Do not duplicate its FK/IK in a
per-robot Python display adapter.

The `aloha/` and `piper/` presets use the same model-loading path for offline
preview and replay. Their display placement does not define hardware calibration
or add a robot deployment recipe. See the [ALOHA](../../../docs/usage/aloha.md)
and [PiPER](../../../docs/usage/piper.md) asset guides.

`base.py`, `yam.py` and the old discovery exports remain for compatibility
consumers, including historical collection-record replay. They are not the live
dashboard integration path. See the [display protocol](../../../docs/development/runtime-config.md#robogui-replay-and-recording).
