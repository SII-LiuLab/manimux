"""Offline geometry for the TacCap UMI follower CAD variant."""

import numpy as np

from manimux.kinematics.base import FloatArray, KinematicCoordinate
from manimux.kinematics.tool import FixedToolGeometry


class TacCapGeometry(FixedToolGeometry):
    """Fixed closed-pad midpoint from the 从夹爪组件0720 CAD.

    The default T_tool_base_tcp is the ``tcp`` in
    assets/end_effectors/umi_follower/end_effector.yaml. It contains no mount.
    ``gripper`` is normalized motor travel: 0 closed, 1 open, not metres or
    a linear pad-distance calibration. Opening is retained in the state but
    does not move this approximate TCP; the real midpoint moves about +/-3 mm.
    This geometry is specific to tahat CAD variant, not every TacCap product.
    A calibrated fixed transform my be supplied explicitly.
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
            tcp_transform = np.eye(4)
            tcp_transform[:3, 3] = [0.11895, 0.0, -0.020995]
        super().__init__(
            tcp_transform,
            coordinates=(KinematicCoordinate("gripper", "normalized"),),
            base_frame=base_frame,
            tcp_frame=tcp_frame,
        )
