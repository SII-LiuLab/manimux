# Third-party notices

## XPolicyLab

`XPolicyLab/` is a Git submodule tracking the `Cuzyoung/XPolicyLab` fork of
`XPolicyLab/XPolicyLab`. It remains a separately versioned project and retains
its own license and third-party notices inside the submodule.

## StarVLA

`StarVLA/` is a separately versioned Git submodule tracking the
`GuHaomingcs/starVLA` fork of `starVLA/starVLA`. The integration is based on upstream
revision `312fac890ab75b7651d2bc4f8f8c8dbb5e055184`; the parent gitlink identifies
its exact version. Source and sampler licenses remain in that submodule.

`manimux/policies/starvla/wire.py` implements StarVLA's native array encoding from
`deployment/model_server/tools/msgpack_numpy.py`. The upstream MIT notice is
retained in `licenses/starvla-MIT.txt`.

## PRM-as-a-Judge

`PRM-as-a-Judge/` is a Git submodule tracking
`YuyangLiu2003/PRM-as-a-Judge`. It remains separately versioned under the
Apache License 2.0; see `PRM-as-a-Judge/LICENSE`.

## i2rt YAM model assets

`manimux/embodiments/arm/yam/assets/i2rt/robot_models/` contains the YAM and linear_4310 model
geometry from `i2rt-robotics/i2rt` at commit
`5d47b358bafb30c65e397f2ece506550a0db4594`. These assets are used only for
robogui rendering and forward/inverse kinematics. The upstream project is under the MIT
License; see `licenses/i2rt-MIT.txt`.

## RoboTwin ALOHA-AgileX follower-arm assets

`manimux/embodiments/arm/aloha_agilex/assets/robotwin/` contains two extracted
follower-arm URDFs and nine DAE meshes from `TianxingChen/RoboTwin2.0`, revision
`d513175169bcdcdb6ddf3de08f10cddc55ad91f1`. The dataset card declares MIT;
the RoboTwin project's notice is retained at
`manimux/embodiments/arm/aloha_agilex/assets/LICENSE.txt`.
Relative mesh references, fixed TCP links and removal of unavailable textures
are documented in the component README. These are offline visual/kinematic
resources, not hardware calibration or a complete mobile-robot simulation.

ARX and PiPER SDKs are separately installed local dependencies. No SDK source
or native library is vendored into the ManiMux package by this integration.

## ARX X5 (2023) kinematic model

`manimux/embodiments/arm/arx_x5/model.urdf` is derived from `ARXroboticsX/X5`
at revision `9a255e38156e3f5a263f341df209ab028d49dad9`, under BSD-3-Clause.
Its notice is retained at `manimux/embodiments/arm/arx_x5/LICENSE.txt`.

This model omits visual/collision meshes while preserving the upstream joint
geometry, inertia and source limits. It does not establish physical calibration
or collision-free motion.

## Standard PiPER model assets

`manimux/embodiments/arm/piper/model.urdf` and ten STL files in
`manimux/embodiments/arm/piper/assets/meshes/` are derived from
`agilexrobotics/piper_ros/src/piper_description` at revision
`ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`, under MIT. Its notice is retained at
`manimux/embodiments/arm/piper/LICENSE.txt`.

Meshes are copied unchanged. The URDF uses relative mesh paths and omits
collision elements; upstream joint geometry, inertia, visual origins and
materials remain unchanged. These resources support offline rendering and
FK/IK, not physical calibration or collision-free motion. Preparation and
model scope are documented in `docs/usage/piper.md`.

## Tianji Marvin model assets

`manimux/embodiments/arm/tianji/assets/` and
`manimux/embodiments/robot/tianji_taccap/assets/` contain the vendor CAD exports of the Tianji
Marvin left arm, right arm and stand (URDF and STL), as bundled in
`SII-LiuLab/universal_robogui` at commit `a7278af`. The stand URDF keeps only
the `Link_Base`/`Link_Stand` links of the vendor's full assembly, and link
colors are replaced with the palette of the bundled YAM model. These assets
are used only for robogui rendering. The DH parameters and joint limits in
`manimux/kinematics/tianji.py` are copied from the vendor SDK's
`CommonConfig/ccs_m6_40.MvKDCfg`.

## Tianji differential IK

The differential velocity QP in `manimux/kinematics/tianji_diff.py` and the
flange Jacobian in `tianji.py` are adapted from `SII-LiuLab/tianji-control`
revision `1e7dfdbc94c62f87501d6485b8c0e43ce6dbf513` (`algos/diff_ik.py`,
`algos/kinematics.py`, and the joint-limit objective in `algos/nullspace.py`).
Tracking-lag guard semantics follow CalibWrist revision
`f17a62a2ab77207de54fdca73dc8ff4d76f87cdb`. See `docs/advanced/tianji-diff-ik.md` for the
port boundaries and numerical validation. The source tianji-control checkout
does not include a project-level license file; no different license is asserted
for this derived code here.

## Marvin SDK

`manimux/embodiments/arm/tianji/sdk/marvin/` contains the Python bindings
(`fx_robot.py`, `fx_kine.py`), Linux x86-64 libraries (`libMarvinSDK.so`,
`libKine.so`) and the `ccs_m6_40.MvKDCfg` arm table from the vendor's
`TJ_FX_ROBOT_CONTRL_SDK`, unmodified. The SDK is Copyright 2025 上海孚晞科技有限公司
under the Apache License 2.0; see the `LICENSE` file in that directory.

## UMI follower gripper assets

`manimux/embodiments/end_effector/taccap/assets/umi_follower/` contains the XenseRobotics
UMI follower gripper CAD (从夹爪组件0720, URDF and STL) with colors replaced by
the bundled YAM gripper palette. Its flange mount and TCP follow the
derivation in `SII-LiuLab/tianji-control` commit `d1def56`.
