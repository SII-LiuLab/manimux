# ALOHA-AgileX assets

ManiMux includes the RoboTwin ALOHA-AgileX follower arms for offline RoboGUI
preview, absolute-joint replay and FK/IK. The bundled model loads without a
robot SDK. The RoboGUI preset has no hardware controller or Home operation.

## Preview and replay

Use a [core ManiMux environment](environments.md), then run from the repository root:

```bash
manimux-robogui --robot aloha --demo --host 127.0.0.1 --port 8087
```

Open `http://127.0.0.1:8087`. Both arms and grippers move using synthetic joint
targets; the overview image and predicted chunks are synthetic too. Omit
`--demo` for a static preview. This demonstration does not run a checkpoint or
simulate physics. The table and dual-arm placement are display context.

To inspect recorded absolute joint targets:

```bash
manimux-robogui --robot aloha --replay-actions actions.npz \
  --action-dt-s 0.03333333333333333 --host 127.0.0.1 --port 8087
```

The NPZ contains `left_arm` and/or `right_arm` arrays of shape `(steps, 7)`:
six joint angles in radians, then normalized opening (0 closed, 1 open).
The replay interval must match the recorded action clock. EEF poses, deltas and
raw model outputs must first pass through the matching policy adapter.
See [offline replay](replay.md) for controls and trajectory requirements.

## Model and source

The follower-arm assets originate from RoboTwin's `arx5_description_isaac.urdf`.
ALOHA names an assembly; its geometry is not a calibrated standard PiPER or
stock ARX X5 model. The [component README](../../manimux/embodiments/arm/aloha_agilex/README.md)
documents the source revision, MIT notice, TCP, visual gripper mapping and
reproducible extraction. It also describes the simulation joint bounds and
numerical IK acceptance criteria.

For physical arms, use the corresponding model and
[ARX/PiPER SDK adapter guide](can-arms.md). SDK installation and device
calibration are separate from loading this asset preset.
