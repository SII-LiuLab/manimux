# Marvin straight gripper visual model

Standalone OmniGripper geometry supplied as `marvin_gripper_description(2)`;
the source package describes it as the Marvin Pro gripper. Installed from
`marvin_gripper_description/urdf/straight_gripper.urdf` and
`mesh/omnigripper/`. The original `package.xml` is retained; it declares MIT,
but the supplied directory contains no separate license text.

The three STL files are unchanged. `end_effector.urdf` changes only ROS
`package://` mesh references to local `meshes/` paths. Original link/joint names,
origins, materials, limits, inertias and the right-finger mimic are preserved.

`end_effector.yaml` follows ManiMux's standalone tool convention:

- Root: `flange`. The URDF already rotates the gripper mount by -90 degrees
  around Z; do not apply that rotation again in the assembly mount.
- One normalized opening input: 0 closed maps to 0.05 m prismatic displacement;
  1 open maps to 0 m. The right finger follows the left via the original mimic.
  These are nominal visual endpoints, not hardware motor calibration.
- TCP: the original `tool0` transform relative to `flange`, composed from the
  source fixed joints. Its translation is zero; it is a reference frame, not
  a calibrated fingertip TCP. The assembly must supply actual mounting and TCP
  calibration before pose-space use.

Load through the existing offline resource API from the repository root:

```python
from manimux.kinematics.end_effector import load_end_effector, attach_end_effector

tool = load_end_effector(
    "manimux/embodiments/end_effector/omnigripper/assets/marvin_straight"
)
closed = tool.joint_positions([0.0])
opened = tool.joint_positions([1.0])
# Once an arm URDF and its flange link are available:
# combined = attach_end_effector(arm_urdf, flange_link, tool)
```

Select this resource with the component option `visual_asset: marvin_straight`.
`OmniGripper.load_model()` defaults to `assets/flange_reference` when the option
is omitted. The RoboGUI preset `flexiv_omnigripper` uses this resource for both
Flexiv arms through `configs/embodiment/robot/flexiv_omnigripper_dual.yaml`.
The Flexiv assembly seats the tool by -6.31 mm along the arm flange Z axis:
the CAD rear plane is at +6.299967 mm, while the wrist mesh ends at -0.01 mm
relative to its flange frame. This makes the two visible planes flush.
A matching +6.31 mm `tcp_transform` translation retains the assembly's existing
flange-reference TCP; the CAD selection does not establish a calibrated grasp TCP.
