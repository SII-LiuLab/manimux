"""Independent grouped TCP kinematics; no hardware reads or command dispatch."""

from collections.abc import Mapping
from types import MappingProxyType

from manimux.kinematics.base import FloatArray, IKResult, ManipulatorKinematicsBase


class RobotKinematics:
    """Group names match RobotState/RobotCommand; each model includes its tool.

    FK/IK may operate on a non-empty subset. Poses remain in each model's named
    base_frame; no shared world frame or inter-arm collision/constraint solve is
    implied. IK returns independent results, never commands or partial execution.
    """

    def __init__(self, models: Mapping[str, ManipulatorKinematicsBase]) -> None:
        if not models or any(not isinstance(k, str) or not k.strip() for k in models):
            raise ValueError("kinematic groups must have non-empty names")
        self._models = MappingProxyType(dict(models))

    @property
    def models(self) -> Mapping[str, ManipulatorKinematicsBase]:
        return self._models

    def _groups(self, values: Mapping) -> None:
        if not values or set(values) - self.models.keys():
            raise ValueError("expected a non-empty subset of configured groups")

    def fk(self, configuration: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        self._groups(configuration)
        return {name: self.models[name].fk(q) for name, q in configuration.items()}

    def ik(
        self,
        targets: Mapping[str, FloatArray],
        seed: Mapping[str, FloatArray],
        *,
        fixed_coordinates: Mapping[str, Mapping[str, float]],
    ) -> dict[str, IKResult]:
        self._groups(targets)
        if set(seed) != set(targets) or set(fixed_coordinates) != set(targets):
            raise ValueError(
                "targets, seeds and fixed-coordinate mappings must have identical groups"
            )
        return {
            name: self.models[name].ik(
                target, seed[name], fixed_coordinates=fixed_coordinates[name]
            )
            for name, target in targets.items()
        }
