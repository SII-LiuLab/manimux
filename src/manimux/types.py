from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
UInt8Array: TypeAlias = NDArray[np.uint8]
GroupVector: TypeAlias = dict[str, FloatArray]
GroupTrajectory: TypeAlias = dict[str, FloatArray]


def _vector_groups(groups: GroupVector, *, label: str) -> GroupVector:
    normalized: GroupVector = {}
    if not groups:
        raise ValueError(f"{label} groups cannot be empty")
    for name, value in groups.items():
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{label} group {name!r} must be non-empty and finite")
        normalized[name] = np.ascontiguousarray(array)
    return normalized


def _trajectory_groups(groups: GroupTrajectory, *, label: str) -> GroupTrajectory:
    normalized: GroupTrajectory = {}
    horizon: int | None = None
    if not groups:
        raise ValueError(f"{label} groups cannot be empty")
    for name, value in groups.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
            raise ValueError(f"{label} group {name!r} must have shape (horizon, dim)")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} group {name!r} must be finite")
        if horizon is None:
            horizon = int(array.shape[0])
        elif horizon != int(array.shape[0]):
            raise ValueError(f"{label} groups must share one horizon")
        normalized[name] = np.ascontiguousarray(array)
    return normalized


@dataclass(slots=True)
class RobotState:
    groups: GroupVector
    monotonic_ns: int
    sequence: int

    def __post_init__(self) -> None:
        self.groups = _vector_groups(self.groups, label="robot state")


@dataclass(slots=True)
class RobotCommand:
    groups: GroupVector
    monotonic_ns: int
    plan_id: str | None

    def __post_init__(self) -> None:
        self.groups = _vector_groups(self.groups, label="robot command")


@dataclass(slots=True)
class SensorFrame:
    name: str
    data: UInt8Array
    capture_monotonic_ns: int
    sequence: int

    def __post_init__(self) -> None:
        array = np.asarray(self.data, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("sensor frame must have shape (height, width, 3)")
        self.data = np.ascontiguousarray(array)


@dataclass(slots=True)
class ObservationSnapshot:
    state: RobotState
    frames: dict[str, SensorFrame] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceRequest:
    session_id: str
    request_seq: int
    observation_time_ns: int
    deadline_ns: int
    observation: ObservationSnapshot
    instruction: str = ""


@dataclass(frozen=True, slots=True)
class ActionContext:
    request_seq: int
    observation_time_ns: int
    created_time_ns: int
    # The first time at which a newly decoded action can be executed.  Runtime
    # adapters may use this to discard source waypoints that are already in the
    # past before doing expensive or failure-prone embodiment conversion (IK).
    execution_time_ns: int | None = None
    # Latest measured state at response handling time.  This is intentionally
    # separate from the observation embedded in the policy request: the latter
    # defines the trajectory clock, while the former is the safest IK seed.
    measured_state: RobotState | None = None
    max_source_steps: int | None = None
    decode_budget_ms: float | None = None
    independent_groups: bool = False


@dataclass(slots=True)
class ActionChunk:
    plan_id: str
    request_seq: int
    observation_time_ns: int
    created_time_ns: int
    action_space: str
    dt_ns: int
    groups: GroupTrajectory
    # Number of leading policy-source rows already removed by the adapter.
    # Keeping this separate preserves observation_time_ns for plan-age checks.
    source_offset_steps: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    # First invalid row per group, relative to this chunk (before timeline trim).
    hold_from_step: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dt_ns <= 0:
            raise ValueError("action chunk dt_ns must be positive")
        if self.source_offset_steps < 0:
            raise ValueError("action chunk source_offset_steps must be non-negative")
        self.groups = _trajectory_groups(self.groups, label="action chunk")
        for name, step in self.hold_from_step.items():
            if name not in self.groups or not 0 <= step < self.horizon_steps:
                raise ValueError("invalid action chunk hold_from_step")

    @property
    def horizon_steps(self) -> int:
        return int(next(iter(self.groups.values())).shape[0])


@dataclass(slots=True)
class InferenceResponse:
    session_id: str
    request_seq: int
    finished_time_ns: int
    inference_ms: float
    raw_action: object | None
    error: str | None = None
    observation_time_ns: int = 0


@dataclass(slots=True)
class ActionHorizon:
    start_time_ns: int
    dt_ns: int
    plan_id: str
    groups: GroupTrajectory
    hold_groups: tuple[str, ...] = ()
    # Unblended joint reference at start_time_ns for measured tracking checks.
    tracking_groups: GroupVector | None = None
    observation_time_ns: int | None = None

    def __post_init__(self) -> None:
        if self.dt_ns <= 0:
            raise ValueError("action horizon dt_ns must be positive")
        self.groups = _trajectory_groups(self.groups, label="action horizon")

    @property
    def horizon_steps(self) -> int:
        return int(next(iter(self.groups.values())).shape[0])


def copy_group_vector(groups: GroupVector) -> GroupVector:
    return {name: value.copy() for name, value in groups.items()}


def copy_action_chunk(chunk: ActionChunk) -> ActionChunk:
    return ActionChunk(
        plan_id=chunk.plan_id,
        request_seq=chunk.request_seq,
        observation_time_ns=chunk.observation_time_ns,
        created_time_ns=chunk.created_time_ns,
        action_space=chunk.action_space,
        dt_ns=chunk.dt_ns,
        groups={name: values.copy() for name, values in chunk.groups.items()},
        source_offset_steps=chunk.source_offset_steps,
        metadata=dict(chunk.metadata),
        hold_from_step=dict(chunk.hold_from_step),
    )
