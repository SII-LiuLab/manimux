# Legacy MolmoAct2 + YAM integration

This directory retains the existing runtime adapter, client, conversion helpers
and legacy model-side code. It is not a template for new learned-model integrations;
those belong in `XPolicyLab/policy/` under the repository's contribution rules.

The direct hardware launcher `manimux-molmoact-yam`, its observer launcher and
`gello_min/env.py` have been retired. They depended on the removed `robots/yam`
implementation and the old left/right hardware configurations.
Use the common runtime's embodiment interface for YAM hardware and the shared
`manimux-camera-server` for cameras. Existing model/runtime adapters remain in place;
this change does not claim a validated XPolicyLab replacement for a checkpoint.

Hardware installation, command semantics and camera ownership are documented in:

- [YAM arm](../../embodiments/arm/yam/README.md)
- [YAM assembly](../../embodiments/robot/yam/README.md)
- [RealSense](../../embodiments/sensor/realsense/README.md)

The component migration and remaining cleanup boundaries are described in
[`docs/yam-integrated-component.md`](../../../../docs/yam-integrated-component.md).
