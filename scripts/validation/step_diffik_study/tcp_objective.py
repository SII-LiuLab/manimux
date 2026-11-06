"""Study-only TCP objective; reuse production QP, constraints and reset semantics.

The historical solver optimizes flange twist after removing the tool offset.
This variant evaluates the same joint velocity QP at the configured TCP.
No production file, joint bound, margin or executor behavior is changed.
"""

import numpy as np

from manimux.kinematics.tianji_diff import (
    DifferentialIKResult,
    TianjiStepDifferentialIK,
)


class TcpFrame:
    def __init__(self, arm, offset):
        self.arm = arm
        self.offset = offset.copy()

    def flange_and_jacobian(self, q):
        # This narrow proxy supplies the QP's optimization frame, despite the
        # inherited method name. Translational Jacobian units remain m/rad.
        flange, jacobian = self.arm.flange_and_jacobian(q)
        lever = flange[:3, :3] @ self.offset[:3, 3]
        tcp_jacobian = jacobian.copy()
        tcp_jacobian[:3] += np.cross(jacobian[3:].T, lever).T
        return flange @ self.offset, tcp_jacobian


class TcpStepDifferentialIK(TianjiStepDifferentialIK):
    def __init__(self, models, config, dt_s):
        super().__init__(models, config, dt_s)
        for name, model in models.items():
            a = model._offset(np.array([0.0]))
            b = model._offset(np.array([1.0]))
            if not np.array_equal(a, b):
                raise ValueError("TCP study requires the recorded fixed tool geometry")
            # Bounds, nullspace cost and margins were initialized from the
            # original arm; only FK/Jacobian evaluation changes below.
            self.solvers[name].kinematics = TcpFrame(model.arm, a)

    def solve(self, name, target_tcp, command_seed, gripper):
        command_seed = np.asarray(command_seed, dtype=float)
        if (
            command_seed.shape != (8,)
            or not np.isfinite(command_seed).all()
            or not np.isfinite(gripper)
            or not 0 <= gripper <= 1
        ):
            return DifferentialIKResult(False, "invalid_input")
        return self.solvers[name].solve(target_tcp, command_seed[:7], self.dt_s)
