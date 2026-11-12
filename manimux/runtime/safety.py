from __future__ import annotations

import math
from enum import StrEnum

import numpy as np

from manimux.types import FloatArray, RobotCommand, RobotState, copy_group_vector


class RuntimeState(StrEnum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAULT = "fault"


# Mink IK and encoder quantization sit ~1 mrad past a joint stop. Abort only
# when the command is a real excursion, not that numerical slop.
_POSITION_LIMIT_TOLERANCE = 1e-3


def _validate_measured_bounds(lower, upper, command_lower, command_upper):
    """Validate an optional independent state box, including its command envelope."""
    if not lower and not upper:
        return
    if (
        not command_lower or not command_upper
        or set(lower) != set(command_lower) or set(upper) != set(command_upper)
    ):
        raise ValueError("measured_position_lower/upper must both match command position groups")
    for name in command_lower:
        lo, hi = np.asarray(lower[name], dtype=float), np.asarray(upper[name], dtype=float)
        cmd_lo, cmd_hi = np.asarray(command_lower[name]), np.asarray(command_upper[name])
        if (
            lo.ndim != 1 or not lo.size or lo.shape != hi.shape
            or lo.shape != cmd_lo.shape or hi.shape != cmd_hi.shape
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo >= hi) or np.any(lo > cmd_lo) or np.any(hi < cmd_hi)
        ):
            raise ValueError(
                f"measured position bounds for {name!r} must be finite, ordered "
                "and enclose command position bounds"
            )


class SafetyGuard:
    def __init__(
        self,
        group_dims: dict[str, int],
        position_limit_abs: float | None,
        *,
        position_lower: dict[str, list[float]] | None = None,
        position_upper: dict[str, list[float]] | None = None,
        max_velocity: dict[str, list[float]] | None = None,
        max_acceleration: dict[str, list[float]] | None = None,
        control_dt_s: float | None = None,
        check_measured_position_limits: bool = True,
        measured_position_lower: dict[str, list[float]] | None = None,
        measured_position_upper: dict[str, list[float]] | None = None,
    ) -> None:
        if not isinstance(check_measured_position_limits, bool):
            raise ValueError("check_measured_position_limits must be boolean")
        self._check_measured_position_limits = check_measured_position_limits
        self._group_dims = dict(group_dims)
        self._position_limit_abs = position_limit_abs
        raw_limits = (position_lower, position_upper, max_velocity)
        configured = [bool(values) for values in raw_limits]
        if (any(configured) or max_acceleration) and not all(configured):
            raise ValueError(
                "command safety position and velocity limits must be configured together"
            )
        self._position_lower = self._normalize_limits("position_lower", position_lower)
        self._position_upper = self._normalize_limits("position_upper", position_upper)
        self._measured_lower = self._normalize_limits("measured_position_lower", measured_position_lower)
        self._measured_upper = self._normalize_limits("measured_position_upper", measured_position_upper)
        _validate_measured_bounds(
            self._measured_lower, self._measured_upper, self._position_lower, self._position_upper,
        )
        self._max_velocity = self._normalize_limits("max_velocity", max_velocity)
        self._max_acceleration = self._normalize_limits("max_acceleration", max_acceleration)
        self._rate_limits_enabled = bool(self._max_velocity)
        if self._rate_limits_enabled and (control_dt_s is None or control_dt_s <= 0):
            raise ValueError("control_dt_s must be positive when command rate limits are set")
        self._control_dt_s = control_dt_s
        self._previous_command: dict[str, FloatArray] | None = None
        self._previous_velocity: dict[str, FloatArray] | None = None

    def set_control_period(self, control_dt_s: float) -> None:
        """Retain command history; express the next rate check at the new period."""
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        self._control_dt_s = control_dt_s

    def _normalize_limits(
        self, name: str, values: dict[str, list[float]] | None
    ) -> dict[str, FloatArray]:
        if not values:
            return {}
        if set(values) != set(self._group_dims):
            raise ValueError(f"{name} groups do not match robot configuration")
        normalized: dict[str, FloatArray] = {}
        for group, dimension in self._group_dims.items():
            vector = np.asarray(values[group], dtype=np.float64)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} group {group!r} is invalid")
            normalized[group] = vector
        return normalized

    def _validate_positions(self, name: str, values: FloatArray, *, label: str) -> None:
        explicit_measured = label == "state" and bool(self._measured_lower)
        lower = self._measured_lower if explicit_measured else self._position_lower
        upper = self._measured_upper if explicit_measured else self._position_upper
        # Explicit state bounds already include their configured allowance.
        tolerance = 0.0 if explicit_measured else _POSITION_LIMIT_TOLERANCE
        if lower:
            below = values < lower[name] - tolerance
            above = values > upper[name] + tolerance
            if np.any(below | above):
                index = int(np.flatnonzero(below | above)[0])
                raise ValueError(
                    f"{label} group {name!r} joint {index} position "
                    f"{values[index]:.6f} is outside "
                    f"[{lower[name][index]:.6f}, {upper[name][index]:.6f}]"
                )

    def validate_state(self, state: RobotState) -> None:
        if set(state.groups) != set(self._group_dims):
            raise ValueError("state groups do not match robot configuration")
        for name, dim in self._group_dims.items():
            values = state.groups[name]
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError(f"state group {name!r} is invalid")
            if self._check_measured_position_limits:
                self._validate_positions(name, values, label="state")

    def reset(self, state: RobotState) -> None:
        """Synchronize the rate checker to an achieved hardware state."""
        self.validate_state(state)
        self._previous_command = copy_group_vector(state.groups)
        self._previous_velocity = {
            name: np.zeros_like(values) for name, values in state.groups.items()
        }

    def validate_command(self, command: RobotCommand) -> None:
        if set(command.groups) != set(self._group_dims):
            raise ValueError("command groups do not match robot configuration")
        candidate_velocity: dict[str, FloatArray] = {}
        for name, dim in self._group_dims.items():
            values = command.groups[name]
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError(f"command group {name!r} is invalid")
            if self._position_limit_abs is not None and np.any(
                np.abs(values) > self._position_limit_abs
            ):
                raise ValueError(f"command group {name!r} exceeds the position limit")
            self._validate_positions(name, values, label="command")
            if not self._rate_limits_enabled:
                continue
            if self._previous_command is None or self._previous_velocity is None:
                raise RuntimeError("command safety rate checker must be reset from measured state")
            assert self._control_dt_s is not None
            velocity = (values - self._previous_command[name]) / self._control_dt_s
            velocity_excess = np.abs(velocity) - self._max_velocity[name]
            if np.any(velocity_excess > 1e-9):
                index = int(np.argmax(velocity_excess))
                raise ValueError(
                    f"command group {name!r} joint {index} velocity "
                    f"{velocity[index]:.6f} exceeds +/-"
                    f"{self._max_velocity[name][index]:.6f}"
                )
            if self._max_acceleration:
                acceleration = (velocity - self._previous_velocity[name]) / self._control_dt_s
                acceleration_excess = np.abs(acceleration) - self._max_acceleration[name]
                if np.any(acceleration_excess > 1e-8):
                    index = int(np.argmax(acceleration_excess))
                    raise ValueError(
                        f"command group {name!r} joint {index} acceleration "
                        f"{acceleration[index]:.6f} exceeds +/-"
                        f"{self._max_acceleration[name][index]:.6f}"
                    )
            candidate_velocity[name] = velocity

        if self._rate_limits_enabled:
            self._previous_command = copy_group_vector(command.groups)
            self._previous_velocity = {
                name: velocity.copy() for name, velocity in candidate_velocity.items()
            }


def command_safety_parameters(**options) -> dict:
    """Fill in module defaults and return an independent dictionary without creating a configuration object."""

    values = {
        "position_lower": {},
        "position_upper": {},
        "max_velocity": {},
        "max_acceleration": {},
        "check_measured_position_limits": True,
        "measured_position_lower": {},
        "measured_position_upper": {},
        **options,
    }
    if not isinstance(values["check_measured_position_limits"], bool):
        raise ValueError("command_safety.check_measured_position_limits must be boolean")
    _validate_measured_bounds(
        values["measured_position_lower"], values["measured_position_upper"],
        values["position_lower"], values["position_upper"],
    )
    values["max_acceleration"] = values["max_acceleration"] or {}
    mappings: tuple[dict[str, list[float]], ...] = (
        values["position_lower"],
        values["position_upper"],
        values["max_velocity"],
    )
    populated = [bool(values) for values in mappings]
    if not any(populated) and (not values["max_acceleration"]):
        return values
    if not all(populated):
        raise ValueError(
            "executor.command_safety requires position_lower, position_upper, "
            "and max_velocity together; max_acceleration is optional"
        )
    if values["max_acceleration"]:
        mappings += (values["max_acceleration"],)
    groups = set(values["position_lower"])
    if not groups or any(set(values) != groups for values in mappings[1:]):
        raise ValueError("executor.command_safety mappings must contain the same groups")
    for group in groups:
        lower = values["position_lower"][group]
        upper = values["position_upper"][group]
        velocity = values["max_velocity"][group]
        acceleration = values["max_acceleration"].get(group, [])
        dimensions = {len(mapping[group]) for mapping in mappings}
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError(
                f"executor.command_safety group {group!r} vectors must share one non-zero dimension"
            )
        samples = lower + upper + velocity + acceleration
        if not all(math.isfinite(value) for value in samples):
            raise ValueError(f"executor.command_safety group {group!r} must be finite")
        if any((lo >= hi for lo, hi in zip(lower, upper, strict=True))):
            raise ValueError(f"executor.command_safety group {group!r} has invalid position bounds")
        if any(value <= 0 for value in velocity + acceleration):
            raise ValueError(
                f"executor.command_safety group {group!r} rate limits must be positive"
            )
    return values
