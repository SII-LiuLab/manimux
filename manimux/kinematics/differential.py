"""Bounded differential steps, distinct from exact-pose IK convergence."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from manimux.kinematics.base import FloatArray


@dataclass(frozen=True, slots=True)
class DifferentialIKResult:
    """An accepted velocity-integrated step need not reach the final pose.

    A rejected step has no candidate. Callers own command holds and tool behavior.
    ``limited`` describes valid bounded progress, not mathematical convergence.
    ``position_bounds`` contains one [lower, upper] pair per coordinate for this
    tick. Check the final executor output against it; smoothing must not bypass
    a measured-tracking or joint-limit constraint.
    """

    accepted: bool
    joints: FloatArray | None = None
    limited: bool = False
    reason: str = ""
    position_bounds: FloatArray | None = None

    def __post_init__(self):
        if self.accepted:
            q = np.array(self.joints, dtype=float, copy=True)
            if q.ndim != 1 or not q.size or not np.isfinite(q).all():
                raise ValueError("accepted differential IK requires finite joint positions")
            q.setflags(write=False)
            object.__setattr__(self, "joints", q)
            bounds = np.array(self.position_bounds, dtype=float, copy=True)
            if (
                bounds.shape != (q.size, 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
                or np.any(q < bounds[:, 0] - 1e-10)
                or np.any(q > bounds[:, 1] + 1e-10)
            ):
                raise ValueError("differential IK requires valid bounds containing its command")
            bounds.setflags(write=False)
            object.__setattr__(self, "position_bounds", bounds)
        elif self.joints is not None or self.position_bounds is not None or not self.reason.strip():
            raise ValueError("rejected differential IK requires a reason and no candidate")


class DifferentialIKSolver(Protocol):
    """Offline, stateful solver using separate command and measured coordinates.

    Transforms use the model's base/TCP, metres and radians. ``dt_s`` is the
    available integration interval; no wall-time backlog is implied. Reset at
    rollout boundaries and when a multi-group step is discarded. This interface
    does not read hardware or choose execution's fallback command.
    """

    def reset(self) -> None: ...

    def step(
        self, target, previous, measured, *, dt_s, fixed_coordinates
    ) -> DifferentialIKResult: ...


class ComposedDifferentialIK:
    """Convert a configured TCP to a flange once, with the requested tool state."""

    def __init__(self, model, arm_solver):
        self.model = model
        self.arm_solver = arm_solver

    def reset(self):
        self.arm_solver.reset()

    def step(self, target, previous, measured, *, dt_s, fixed_coordinates):
        model = self.model
        previous = model._configuration(previous)
        measured = model._configuration(measured)
        arm_size = model.arm.num_arm_joints
        tool_names = [c.name for c in model.coordinates[arm_size:]]
        if set(fixed_coordinates) != set(tool_names):
            raise NotImplementedError("differential IK requires all tool coordinates fixed")
        candidate = previous.copy()
        candidate[arm_size:] = [fixed_coordinates[name] for name in tool_names]
        candidate = model._configuration(candidate)
        target_flange = model.flange_target(target, candidate[arm_size:])
        result = self.arm_solver.step(
            target_flange, previous[:arm_size], measured[:arm_size], dt_s=dt_s
        )
        if not result.accepted:
            return result
        if result.joints.shape != (arm_size,):
            raise ValueError("differential flange solver returned an invalid joint layout")
        candidate[:arm_size] = result.joints
        tool_bounds = [
            [0.0, 1.0] if c.unit == "normalized" else [value, value]
            for c, value in zip(model.coordinates[arm_size:], candidate[arm_size:], strict=True)
        ]
        bounds = np.vstack((result.position_bounds, np.asarray(tool_bounds).reshape(-1, 2)))
        return DifferentialIKResult(True, candidate, result.limited, result.reason, bounds)
