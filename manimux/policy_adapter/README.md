# Policy adapters

This package owns observation preparation and conversion of policy-service actions
into ManiMux `ActionChunk`s. Experiments select an actual Python implementation;
there is no model-name registry or forwarding package in `integrations/`.

```yaml
policy:
  worker: xpolicylab_ws
  adapter:
    type: manimux.policy_adapter.joint:JointAdapter
    group_order: [left_arm, right_arm]
    group_prefixes: {left_arm: left, right_arm: right}
    gripper_dofs: 1
    camera_map:
      cam_head: front_camera
      cam_left_wrist: left_camera
      cam_right_wrist: right_camera
  action_dt_s: 0.03333333333333333
  horizon_steps: 50
  options:
    server: ws://127.0.0.1:8500
```

`policy.adapter` holds semantic mappings; `policy.options` holds transport/backend
options. Timing and horizon are declared once in `policy`. The wire codec reads the
same camera/group mapping when serializing a request.

The factory calls `Adapter(robot, policy, kinematics=...)`. The runtime can provide
its assembled robot's offline kinematics. Decoder processes never receive live
hardware handles. Specialized solvers and their configured tolerances are preserved
by this directory migration; consolidating them must preserve their TCP/IK behavior.

- `base.py`: one interface, identity observation/request hooks and abstract
  `decode_action(raw, context)`. The base assumes no joint action format.
- `joint.py`: absolute joint dictionaries to grouped trajectories. Arm values
  precede gripper values; units remain unchanged. No normalization, delta recovery,
  IK or smoothing is performed here.
- `sapolicy/`, `openwam/`, `umi_dp/`, `xr1/`, `lingbot_vla2/`: specialized pose,
  observation history and action conversion implementations.
- `abc_yam.py`, `molmoact_yam.py`: existing native-backend matrix decoders. Their
  relocation does not migrate or validate those legacy model servers.

One model may select multiple adapters when its service exposes different contracts.
Multiple models may share an adapter when their contracts agree. A new checkpoint
alone does not need a new class. Put mappings in configuration and conversion
procedures in Python, documenting units, coordinate frames, delta anchors and time.

Model loading, normalization and model sampling remain in XPolicyLab. Transport
remains in `policies/`; scheduling in `runtime/`; command generation in executors.
No empty generic TCP adapter is provided: existing pose adapters differ in anchor,
quaternion, timing and IK behavior and cannot be merged solely because they use poses.
