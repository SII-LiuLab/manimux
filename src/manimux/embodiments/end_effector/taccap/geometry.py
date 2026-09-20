"""Offline geometry for the TacCap UMI follower CAD variant."""

from pathlib import Path

import yaml

from manimux.kinematics.base import FloatArray, KinematicCoordinate, transform_from_xyz_rpy
from manimux.kinematics.composed import FixedToolGeometry

ASSET_DIRECTORY = Path(__file__).parent / "assets" / "umi_follower"


class TacCapGeometry(FixedToolGeometry):
    """Fixed closed-pad midpoint from the 从夹爪组件0720 CAD.

    The default T_tool_base_tcp is the ``tcp`` in
    embodiments/end_effector/taccap/assets/umi_follower/end_effector.yaml. It contains no mount.
    ``gripper`` is normalized motor travel: 0 closed, 1 open, not metres or
    a linear pad-distance calibration. Opening is retained in the state but
    does not move this approximate TCP; the real midpoint moves about +/-3 mm.
    This geometry is specific to that CAD variant, not every TacCap product.
    A calibrated fixed transform may be supplied explicitly.
    No TacCap SDK is imported or device opened.
    """

    def __init__(
        self,
        *,
        tcp_transform: FloatArray | None = None,
        base_frame: str = "taccap_base",
        tcp_frame: str = "taccap_tcp",
    ) -> None:
        if tcp_transform is None:
            # 控制几何和 Viewer 共用资源中的 TCP，不在 Python 中维护第二套数值。
            spec = yaml.safe_load((ASSET_DIRECTORY / "end_effector.yaml").read_text())
            tcp_transform = transform_from_xyz_rpy(**spec["tcp"], name="TacCap TCP")
        super().__init__(
            tcp_transform,
            coordinates=(KinematicCoordinate("gripper", "normalized"),),
            base_frame=base_frame,
            tcp_frame=tcp_frame,
        )
