# Standard PiPER assets

ManiMux includes the standard AgileX PiPER arm and two-finger gripper for
offline RoboGUI preview, absolute-joint replay and FK/IK. The PiPER component
uses one URDF for both rendering and kinematics. Loading it does not import
pyAgxArm, open CAN, enable motors or connect cameras.

This preset covers **standard PiPER**, with the official post-S-V1.6-3 joint
geometry. PiPER H, L, X, older joint-frame definitions and other grippers are
outside its scope. See the [official version note](https://github.com/agilexrobotics/piper_ros/blob/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/README.MD#0-%E6%B3%A8%E6%84%8Furdf%E7%89%88%E6%9C%AC).
Device firmware selection alone does not change the offline model's geometry.

## Preview and replay

Use an environment containing ManiMux's core dependencies, such as the
[runtime environment](robot-sdks.md#runtime-environment). From the repository root:

```bash
manimux-robogui --robot piper --demo \
  --host 127.0.0.1 --port 8087
```

Open `http://127.0.0.1:8087`. Both arms and their grippers move using synthetic
joint targets; the overview image and predicted chunks are synthetic too.
This is a visual demonstration, not checkpoint inference or robot execution.
The dual-arm placement and table are display context, not calibrated mounting.

To inspect an existing absolute-joint trajectory:

```bash
manimux-robogui --robot piper \
  --replay-actions actions.npz --action-dt-s 0.03333333333333333 \
  --host 127.0.0.1 --port 8087
```

The NPZ contains `left_arm` and/or `right_arm` arrays of shape `(steps, 7)`:
six joint angles in radians, then normalized gripper opening (0 closed, 1 open).
The nominal visual finger coordinates are `0.035 * opening` and
`-0.035 * opening` metres. Physical width endpoints remain station calibration.
EEF poses, deltas and raw normalized model outputs must first pass through
the matching policy adapter. See [offline replay](replay.md) for controls and
trajectory requirements.

The TCP remains the `link6` flange in `base_link`; the gripper's mesh does not
define a new grasp-centre frame. The preset does not add hardware Home or a
validated policy deployment. For device configuration and current limitations,
see the [PiPER SDK adapter guide](can-arms.md).

## Source and regeneration

The URDF and all ten STL meshes come from
[agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros/tree/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper_description),
revision `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`. The upstream MIT notice
is retained in `manimux/embodiments/arm/piper/LICENSE.txt` and recorded in the
repository's [third-party notices](../../licenses/THIRD_PARTY_NOTICES.md).

The preparation script copies meshes unchanged, rewrites ROS package paths
relative to `model.urdf`, and removes collision elements. It preserves upstream
joint geometry, inertia, visual origins, materials and exporter comments.
Collision checking and a complete physics simulation are not provided by this
visual preset. Packaged paths are portable; no external ROS package or local
source checkout is needed to use the result.

To regenerate from the same source, keep the upstream checkout in the ignored
SDK workspace:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/agilexrobotics/piper_ros.git envs/aloha/sdk/piper_ros
git -C envs/aloha/sdk/piper_ros sparse-checkout set src/piper_description
git -C envs/aloha/sdk/piper_ros checkout ac41fcbcdda598f01b51cf6175ed9a24d0dacadc
envs/aloha/.venv/bin/python scripts/assets/prepare_piper.py \
  --upstream envs/aloha/sdk/piper_ros
```

Offline validation covers every real mesh after relocation, visual joint
mapping, rendering/FK agreement, actual RoboGUI loading, synthetic targets,
replay seeking and wheel packaging. It does not establish physical calibration
or real-robot readiness.
