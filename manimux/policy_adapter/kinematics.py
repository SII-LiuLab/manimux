"""Pose adapters use the configured robot model, including in decoder processes."""

import numpy as np

from manimux.policy_adapter.base import PolicyAdapter


class KinematicAdapter(PolicyAdapter):
    """Common access for pose adapters with a trailing scalar tool coordinate."""

    # Adapters that route chunk IK through _solve_ik may select
    # policy.adapter.ik_backend: mink (default) | dls.
    supports_dls_ik = False

    def __init__(self, robot, policy, *, kinematics=None):
        if kinematics is None:
            from manimux.embodiments.robot.base import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        super().__init__(robot, policy, kinematics=kinematics)
        if "kinematics" in policy["adapter"] or "kinematics_options" in policy["adapter"]:
            raise ValueError("kinematics belongs in robot.config")
        options = policy["adapter"]
        self.ik_backend = options.get("ik_backend", "mink")
        if self.ik_backend not in ({"mink", "dls"} if self.supports_dls_ik else {"mink"}):
            raise ValueError(f"{type(self).__name__} does not support ik_backend: {self.ik_backend}")
        if self.ik_backend == "mink" and options.get("dls_ik"):
            raise ValueError("policy.adapter.dls_ik applies only to ik_backend: dls")
        self.dls_solvers = {}
        if self.ik_backend == "dls":
            from manimux.kinematics.dls import DLSIK, DLSIKConfig

            config = DLSIKConfig.model_validate(options.get("dls_ik", {}))
            self.dls_solvers = {
                name: DLSIK(model, config) for name, model in self.kinematics.models.items()
            }

    def ik_sequence(self, group, seed, *, approximate=None):
        # DLS accepts bounded residuals; record them in the ik metadata.
        approximate = bool(self.dls_solvers) if approximate is None else approximate
        return super().ik_sequence(group, seed, approximate=approximate)

    def _fk(self, group, joints, gripper):
        return self.kinematics.models[group].fk(np.r_[joints, gripper])

    def _ik(self, group, target, joints, gripper):
        model = self.kinematics.models[group]
        result = model.ik(
            target,
            np.r_[joints, gripper],
            fixed_coordinates={model.coordinates[-1].name: gripper},
        )
        # The shared execution policy decides how to handle nonconvergence.
        solved = result.joints[: len(joints)] if result.converged else np.asarray(joints).copy()
        return result.converged, solved

    def _solve_ik(self, group, target, joints, gripper):
        """(converged, joints, residual); residual is set for an accepted DLS approximation."""
        solver = self.dls_solvers.get(group)
        if solver is None:
            return (*self._ik(group, target, joints, gripper), None)
        return solver.solve(target, joints, gripper)
