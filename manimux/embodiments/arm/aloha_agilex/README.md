# RoboTwin ALOHA-AgileX follower-arm model

This component provides offline geometry for the `aloha_agilex` embodiment used
by the RoboTwin recipes and standalone RoboGUI. It loads through `RobotModel` and the generic
`RobotView`; no vendor SDK is needed to view or replay actions.

Each arm has six revolute coordinates in radians followed by `gripper`, a
normalized opening: 0 closed, 1 open. The source joint order is `fl_joint1..6`
on the left and `fr_joint1..6` on the right. Display expands the last coordinate
into two finger displacements using `-0.01 + 0.055 * opening` metres, matching
the upstream recipe. This deliberately follows the recipe rather than the
URDF's nominal finger limits. The fixed TCP is 0.12 m along link6's x axis,
with link6's orientation; it does not apply RoboTwin's policy-specific EEF
rotation convention.

`AlohaAgilexKinematics` uses the same URDF for FK and numerical IK. IK honours
named fixed coordinates and the source simulation joint bounds, and accepts a
result only below 0.1 mm position and 0.0001 rad orientation error. Rejection
returns no candidate joints. The URDF's broad arm limits are simulation values,
not real-device motion limits or evidence of a collision-free path.

The included assembly has no hardware factory or SDK controller. Its geometry
must not be substituted for a calibrated ARX X5 or PiPER model. See the
[ALOHA asset guide](../../../../docs/usage/aloha.md) for preview and replay.
Physical arm setup is covered separately by the
[SDK adapter guide](../../../../docs/usage/can-arms.md).

## Asset source and modifications

Source: [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/d513175169bcdcdb6ddf3de08f10cddc55ad91f1/embodiments/aloha-agilex),
revision `d513175169bcdcdb6ddf3de08f10cddc55ad91f1`. Its dataset card declares
MIT; the RoboTwin project's MIT notice is retained in `assets/LICENSE.txt`.

Included: two follower-arm URDFs and nine shared DAE meshes. The extraction:

- Keeps the follower arms' joint origins, axes, inertias and mesh geometry.
- Uses relative mesh paths and adds the explicitly named fixed TCP.
- Omits the mobile base, leader arms, cameras and collision elements.
- Removes references to unavailable `Image_269.png` and `Image_290.png` textures
  and replaces their material channels with neutral grey. Other colours remain.

Regenerate from the original downloaded `aloha-agilex` folder, from the repository root:

```bash
python scripts/assets/prepare_aloha_agilex.py --source /path/to/aloha-agilex
```

The generated assets are for rendering and offline kinematics. They are not a
complete ALOHA mobile-robot simulation.
