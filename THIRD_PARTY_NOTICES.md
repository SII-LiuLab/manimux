# Third-party notices

## XPolicyLab

`XPolicyLab/` is a Git submodule tracking the `Cuzyoung/XPolicyLab` fork of
`XPolicyLab/XPolicyLab`. It remains a separately versioned project and retains
its own license and third-party notices inside the submodule.

## PRM-as-a-Judge

`PRM-as-a-Judge/` is a Git submodule tracking
`YuyangLiu2003/PRM-as-a-Judge`. It remains separately versioned under the
Apache License 2.0; see `PRM-as-a-Judge/LICENSE`.

## MolmoAct2

`src/manimux/integrations/molmoact_yam/` is derived from the YAM evaluation
example in `allenai/molmoact2` at commit
`3d15be665dec918d6b3fe0cf606284adc4677bba`. ManiMux reorganizes it as an
internal package and adds asynchronous rollout observation, viewer events,
runtime configuration, and shutdown behavior. The upstream project is under
Apache License 2.0; see `licenses/MolmoAct2-APACHE-2.0.txt`.

## i2rt YAM model assets

`src/manimux/assets/i2rt/robot_models/` contains the YAM and linear_4310 model
geometry from `i2rt-robotics/i2rt` at commit
`5d47b358bafb30c65e397f2ece506550a0db4594`. These assets are used only for
viewer rendering and forward kinematics. The upstream project is under the MIT
License; see `licenses/i2rt-MIT.txt`.

## Tianji Marvin model assets

`src/manimux/assets/tianji/` contains the vendor CAD exports of the Tianji
Marvin left arm, right arm and stand (URDF and STL), as bundled in
`SII-LiuLab/universal_viewer` at commit `a7278af`. The stand URDF keeps only
the `Link_Base`/`Link_Stand` links of the vendor's full assembly, and link
colors are replaced with the palette of the bundled YAM model. These assets
are used only for viewer rendering. The DH parameters and joint limits in
`src/manimux/kinematics/tianji.py` are copied from the vendor SDK's
`CommonConfig/ccs_m6_40.MvKDCfg`.

## Marvin SDK

`src/manimux/robots/tianji/vendor/marvin/` contains the Python bindings
(`fx_robot.py`, `fx_kine.py`), Linux x86-64 libraries (`libMarvinSDK.so`,
`libKine.so`) and the `ccs_m6_40.MvKDCfg` arm table from the vendor's
`TJ_FX_ROBOT_CONTRL_SDK`, unmodified. The SDK is Copyright 2025 上海孚晞科技有限公司
under the Apache License 2.0; see the `LICENSE` file in that directory.

## UMI follower gripper assets

`src/manimux/assets/end_effectors/umi_follower/` contains the XenseRobotics
UMI follower gripper CAD (从夹爪组件0720, URDF and STL) with colors replaced by
the bundled YAM gripper palette. Its flange mount and TCP follow the
derivation in `SII-LiuLab/tianji-control` commit `d1def56`.
