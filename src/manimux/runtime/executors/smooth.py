from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from manimux.config import SmoothConfig
from manimux.runtime.executors.limits import (
    ScalarLimits,
    decelerate_velocity,
    limit_step,
    limit_velocity,
    tracking_step,
)
from manimux.types import (
    ActionHorizon,
    GroupVector,
    RobotCommand,
    RobotState,
    copy_group_vector,
)


@dataclass
class _GripperEvent:
    target: np.ndarray
    plan_id: str
    kind: str = "release"
    phase: str = "approach"
    stable_since_ns: int | None = None
    stable_aperture: float | None = None
    completed_ns: int | None = None
    phase_started_ns: int | None = None
    failed_ns: int | None = None
    failure_reason: str | None = None


class SmoothExecutor:
    def __init__(self, config: SmoothConfig, control_dt_s: float) -> None:
        self._dt_s = control_dt_s
        rc = 1.0 / (2.0 * math.pi * config.cutoff_hz)
        self._alpha = control_dt_s / (rc + control_dt_s)
        self._limits = ScalarLimits(
            max_velocity=config.max_velocity,
            max_acceleration=config.max_acceleration,
            position_limit_abs=config.position_limit_abs,
        )
        self._gripper = config.gripper
        self._release_guard = config.release_guard
        self._grasp_guard = config.grasp_guard
        self._tracking_kinematics = None
        if config.release_guard is not None:
            from manimux.kinematics import build_kinematics

            self._tracking_kinematics = build_kinematics(
                config.release_guard.kinematics, **config.release_guard.kinematics_options
            )
        self.braking_tracking = config.tracking_mode == "braking"
        self._previous: GroupVector | None = None
        self._previous_velocity: GroupVector | None = None
        self._gripper_closed: dict[str, bool] = {}
        self._gripper_closed_since_ns: dict[str, int | None] = {}
        self._gripper_open_candidate_ns: dict[str, int | None] = {}
        self.gripper_diagnostics: dict[str, dict[str, object]] = {}
        self._gripper_events: dict[str, _GripperEvent] = {}
        self._release_armed: dict[str, bool] = {}
        self._grasp_armed: dict[str, bool] = {}
        self._grasp_bypassed: set[str] = set()
        self._release_bypassed: set[str] = set()

    @property
    def has_pending_release(self) -> bool:
        return any(event.kind == "release" for event in self._gripper_events.values())

    @property
    def has_pending_gripper_event(self) -> bool:
        return bool(self._gripper_events)

    @property
    def horizon_steps(self) -> int:
        return 2

    def brake_hold(self, now_ns: int, state: RobotState) -> RobotCommand:
        """On a RUNNING inference gap, decelerate rather than jump to feedback.

        Freeze the gripper command. Pause/Finish retain their explicit runtime
        behavior; this method is only for an exhausted action timeline.
        """
        if self._previous is None or self._previous_velocity is None:
            self.reset(state)
        assert self._previous is not None and self._previous_velocity is not None
        if self.has_pending_gripper_event:
            # A validated gripper-event target survives ordinary inference gaps. Other
            # groups still brake; Pause/Home call reset and cancel pending events.
            reference = ActionHorizon(
                now_ns, int(self._dt_s * 1e9), "gripper-gap",
                {name: np.tile(value, (2, 1)) for name, value in self._previous.items()},
                hold_groups=tuple(self._previous),
                tracking_groups=copy_group_vector(self._previous),
            )
            command = self.step(now_ns, state, reference)
            command.plan_id = None
            return command
        for name, velocity in self._previous_velocity.items():
            next_velocity = decelerate_velocity(
                velocity, self._limits.max_acceleration, self._dt_s
            )
            if self._gripper is not None and name in self._gripper.group_indices:
                next_velocity[self._gripper.group_indices[name]] = 0.0
            self._previous[name] += next_velocity * self._dt_s
            self._previous_velocity[name] = next_velocity
        return RobotCommand(copy_group_vector(self._previous), now_ns, plan_id=None)

    def reset(self, state: RobotState) -> None:
        self._previous = copy_group_vector(state.groups)
        self._previous_velocity = {
            name: np.zeros_like(value) for name, value in state.groups.items()
        }
        self._gripper_closed = {}
        self._gripper_closed_since_ns = {}
        self._gripper_open_candidate_ns = {}
        self.gripper_diagnostics = {}
        self._gripper_events = {}
        self._release_armed = {}
        self._grasp_armed = {}
        self._grasp_bypassed = set()
        self._release_bypassed = set()
        if self._gripper is None:
            return
        for name, index in self._gripper.group_indices.items():
            if name not in state.groups:
                raise ValueError(f"smooth gripper group {name!r} is absent from robot state")
            if index >= len(state.groups[name]):
                raise ValueError(
                    f"smooth gripper index {index} is outside group {name!r} "
                    f"with dimension {len(state.groups[name])}"
                )
            closed = bool(state.groups[name][index] <= self._gripper.close_threshold)
            self._gripper_closed[name] = closed
            self._release_armed[name] = bool(
                state.groups[name][index] <= self._gripper.close_threshold
            )
            self._grasp_armed[name] = bool(
                state.groups[name][index] >= self._gripper.open_threshold
            )
            self._gripper_closed_since_ns[name] = state.monotonic_ns if closed else None
            self._gripper_open_candidate_ns[name] = None

    def _shape_grippers(
        self,
        now_ns: int,
        reference: ActionHorizon,
        output: GroupVector,
        velocities: GroupVector,
        state: RobotState,
    ) -> None:
        if self._gripper is None:
            return
        assert self._previous is not None
        assert self._previous_velocity is not None
        min_closed_ns = int(self._gripper.min_closed_s * 1_000_000_000)
        open_confirm_ns = int(self._gripper.open_confirm_s * 1_000_000_000)
        for name, index in self._gripper.group_indices.items():
            if name in reference.hold_groups:
                output[name][index] = self._previous[name][index]
                velocities[name][index] = 0.0
                event = self._gripper_events.get(name)
                if event is None or event.phase != "await_replan":
                    self.gripper_diagnostics[name] = {
                        "release_blocked": True, "reason": "ik_hold",
                    }
                continue
            desired = float(reference.groups[name][0, index])
            event = self._gripper_events.get(name)
            if event is not None:
                if event.phase == "approach":
                    output[name][index] = self._previous[name][index]
                    velocities[name][index] = 0.0
                    continue
                desired = float(event.target[index])
            closed = self._gripper_closed[name]
            if self._gripper.mode == "continuous":
                goal = float(
                    np.clip(
                        desired,
                        self._gripper.closed_value,
                        self._gripper.open_value,
                    )
                )
            else:
                if not closed and desired <= self._gripper.close_threshold:
                    closed = True
                    self._gripper_closed[name] = True
                    self._gripper_closed_since_ns[name] = now_ns
                    self._gripper_open_candidate_ns[name] = None
                elif closed:
                    closed_since = self._gripper_closed_since_ns[name]
                    hold_elapsed = (
                        closed_since is not None
                        and now_ns - closed_since >= min_closed_ns
                    )
                    if hold_elapsed and desired >= self._gripper.open_threshold:
                        candidate = self._gripper_open_candidate_ns[name]
                        if candidate is None:
                            self._gripper_open_candidate_ns[name] = now_ns
                        elif now_ns - candidate >= open_confirm_ns:
                            closed = False
                            self._gripper_closed[name] = False
                            self._gripper_closed_since_ns[name] = None
                            self._gripper_open_candidate_ns[name] = None
                    else:
                        self._gripper_open_candidate_ns[name] = None

                goal = (
                    self._gripper.closed_value if closed else self._gripper.open_value
                )
            previous = float(self._previous[name][index])
            previous_velocity = float(self._previous_velocity[name][index])
            velocity = float(
                limit_velocity(
                    np.asarray((goal - previous) / self._dt_s),
                    np.asarray(previous_velocity),
                    self._dt_s,
                    self._gripper.max_velocity,
                    self._gripper.max_acceleration,
                    self._gripper.max_closing_velocity,
                )
            )
            command = previous + velocity * self._dt_s
            output[name][index] = float(
                np.clip(
                    command,
                    self._gripper.closed_value,
                    self._gripper.open_value,
                )
            )
            velocities[name][index] = velocity

    @staticmethod
    def _rotation_error(actual: np.ndarray, target: np.ndarray) -> float:
        cosine = (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _gripper_reference(
        self, now_ns: int, state: RobotState, reference: ActionHorizon,
    ) -> ActionHorizon:
        """Finish a grasp/release at its captured pose, then await a fresh observation."""
        if self._release_guard is None:
            return reference
        assert self._gripper is not None and self._previous is not None
        if reference.tracking_groups is None:
            raise ValueError("latched release requires an unblended tracking reference")
        groups = {name: rows.copy() for name, rows in reference.groups.items()}
        tracking = copy_group_vector(reference.tracking_groups)
        holds = set(reference.hold_groups)
        kin = self._tracking_kinematics
        for name, index in self._gripper.group_indices.items():
            measured = state.groups[name]
            if index != kin.num_arm_joints or len(measured) != kin.state_dim:
                raise ValueError(
                    "gripper pose tracking requires packed arm joints followed by gripper"
                )
            event = self._gripper_events.get(name)
            if (event is not None and event.phase == "await_replan"
                    and reference.observation_time_ns is not None
                    and reference.observation_time_ns > event.failed_ns
                    and name not in holds):
                # A timeout is not a completed event. Resume only from a valid
                # post-timeout observation, without latching this arm again.
                del self._gripper_events[name]
                if event.kind == "grasp":
                    self._grasp_armed[name] = False
                else:
                    self._release_armed[name] = False
                event = None
            if (event is not None and event.phase == "await_observation"
                    and reference.observation_time_ns is not None
                    and reference.observation_time_ns > event.completed_ns):
                del self._gripper_events[name]
                self._release_armed[name] = event.kind == "grasp"
                self._grasp_armed[name] = event.kind == "release"
                event = None
            if event is None:
                if self._previous[name][index] <= self._gripper.close_threshold:
                    self._release_armed[name] = True
                if (self._previous[name][index] >= self._gripper.open_threshold
                        and measured[index] >= self._gripper.open_threshold):
                    self._grasp_armed[name] = True
                signal = float(tracking[name][index])
                if (self._release_armed.get(name, False) and name not in holds
                        and name not in self._release_bypassed
                        and signal >= self._gripper.open_threshold):
                    target = tracking[name].copy()
                    target[index] = self._gripper.open_value
                    event = _GripperEvent(
                        target, reference.plan_id, phase_started_ns=int(now_ns)
                    )
                    self._gripper_events[name] = event
                    self._release_armed[name] = False
                elif (self._grasp_guard is not None and self._grasp_armed.get(name, False)
                        and name not in self._grasp_bypassed
                        and name not in holds and signal < self._gripper.open_threshold):
                    # Capture closure onset, not the late fully-closed waypoint
                    # that may already belong to the model's lifting trajectory.
                    target = tracking[name].copy()
                    target[index] = self._gripper.closed_value
                    event = _GripperEvent(
                        target, reference.plan_id, kind="grasp", phase_started_ns=int(now_ns)
                    )
                    self._gripper_events[name] = event
                    self._grasp_armed[name] = False
            if event is None:
                self.gripper_diagnostics[name] = {
                    "release_phase": "idle", "release_blocked": False, "grasp_phase": "idle",
                    "grasp_guard_bypassed": name in self._grasp_bypassed,
                    "release_guard_bypassed": name in self._release_bypassed,
                    "desired_aperture": float(reference.groups[name][0, index]),
                }
                continue
            actual_pose = kin.fk(measured[:index], float(measured[index]))
            target_pose = kin.fk(event.target[:index], float(event.target[index]))
            error = float(np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3]))
            rotation_error = None
            if event.kind == "grasp":
                guard = self._grasp_guard
                assert guard is not None
                command_pose = kin.fk(self._previous[name][:index],
                                      float(self._previous[name][index]))
                rotation_error = self._rotation_error(actual_pose, target_pose)
                arrived = (
                    error <= guard.position_tolerance_m
                    and rotation_error <= guard.rotation_tolerance_rad
                    and np.linalg.norm(command_pose[:3, 3] - target_pose[:3, 3])
                    <= guard.position_tolerance_m
                    and self._rotation_error(command_pose, target_pose)
                    <= guard.rotation_tolerance_rad
                    and np.max(np.abs(self._previous_velocity[name][:index])) <= 0.1
                )
                if event.phase == "approach" and arrived:
                    event.phase = "closing"
                    event.phase_started_ns = int(now_ns)
                if event.phase == "closing":
                    # A bottle prevents the measured aperture reaching zero.
                    # Wait for the close command and stable measured aperture;
                    # this confirms motion completion, not object detection.
                    ready = (
                        arrived
                        and self._previous[name][index] <= self._gripper.closed_value + .02
                        and measured[index] < self._gripper.open_threshold - .05
                    )
                    aperture = float(measured[index])
                    if not ready:
                        event.stable_since_ns = None
                        event.stable_aperture = None
                    elif (event.stable_aperture is None
                          or abs(aperture - event.stable_aperture) > guard.aperture_stability):
                        event.stable_since_ns = int(now_ns)
                        event.stable_aperture = aperture
                    elif now_ns - event.stable_since_ns >= guard.settle_s * 1e9:
                        event.phase = "await_observation"
                        event.completed_ns = int(now_ns)
                timeout_s = guard.phase_timeout_s
            else:
                if event.phase == "approach" and error <= self._release_guard.position_tolerance_m:
                    event.phase = "opening"
                    event.phase_started_ns = int(now_ns)
                if (event.phase == "opening"
                        and self._previous[name][index] >= self._gripper.open_value - .02
                        and measured[index] >= self._gripper.open_value - .05):
                    event.phase = "await_observation"
                    event.completed_ns = int(now_ns)
                timeout_s = self._release_guard.phase_timeout_s
            if (event.phase in {"approach", "closing", "opening"}
                    and event.phase_started_ns is not None
                    and now_ns - event.phase_started_ns >= timeout_s * 1e9):
                event.failure_reason = f"{event.phase}_timeout"
                event.failed_ns = int(now_ns)
                event.phase = "await_replan"
                bypassed = self._grasp_bypassed if event.kind == "grasp" else self._release_bypassed
                bypassed.add(name)
            # Keep a static arm target (zero reference velocity) through the event.
            if event.phase == "await_replan":
                # Brake existing command velocity and freeze aperture while
                # waiting for fresh inference, rather than continuing the old pose.
                groups[name][:] = self._previous[name]
                tracking[name] = self._previous[name].copy()
                holds.add(name)
            else:
                groups[name][:] = event.target
                tracking[name] = event.target.copy()
                holds.discard(name)
            self.gripper_diagnostics[name] = {
                "release_phase": event.phase if event.kind == "release" else "idle",
                "grasp_phase": event.phase if event.kind == "grasp" else "idle",
                "release_blocked": event.kind == "release" and event.phase == "approach",
                "grasp_blocked": event.kind == "grasp" and event.phase == "approach",
                "event_kind": event.kind,
                f"{event.kind}_plan_id": event.plan_id,
                f"{event.kind}_target": event.target.tolist(),
                "tracking_position_error_m": error,
                "tracking_rotation_error_rad": rotation_error,
                "desired_aperture": float(reference.groups[name][0, index]),
                "completed_ns": event.completed_ns,
                "failed_ns": event.failed_ns,
                "failure_reason": event.failure_reason,
                "grasp_guard_bypassed": name in self._grasp_bypassed,
                "release_guard_bypassed": name in self._release_bypassed,
            }
        return ActionHorizon(
            reference.start_time_ns, reference.dt_ns, reference.plan_id, groups,
            hold_groups=tuple(sorted(holds)), tracking_groups=tracking,
            observation_time_ns=reference.observation_time_ns,
        )

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand:
        if self._previous is None or self._previous_velocity is None:
            self.reset(state)
        assert self._previous is not None
        assert self._previous_velocity is not None
        reference = self._gripper_reference(now_ns, state, reference)
        if reference.hold_groups and not self.braking_tracking:
            raise ValueError("per-group hold requires braking tracking")
        if self.braking_tracking:
            target = {name: values[0] for name, values in reference.groups.items()}
            target_velocity = {
                name: (
                    (values[1] - values[0]) / (reference.dt_ns / 1e9)
                    if len(values) > 1 else np.zeros_like(values[0])
                )
                for name, values in reference.groups.items()
            }
            output, velocities = {}, {}
            for name in target:
                limits = self._limits
                approach_limited = False
                if (self._grasp_guard is not None
                        and self._grasp_guard.approach_max_velocity is not None
                        and self._gripper is not None
                        and name in self._gripper.group_indices):
                    event = self._gripper_events.get(name)
                    index = self._gripper.group_indices[name]
                    approach_limited = (
                        event is not None and event.kind == "grasp"
                        and event.phase in {"approach", "closing"}
                    ) or (
                        event is None
                        and self._previous[name][index] > self._gripper.close_threshold
                    )
                    if approach_limited:
                        approach_velocity = self._grasp_guard.approach_max_velocity
                        limits = ScalarLimits(
                            approach_velocity if limits.max_velocity is None else min(
                                limits.max_velocity, approach_velocity
                            ),
                            limits.max_acceleration, limits.position_limit_abs,
                        )
                    self.gripper_diagnostics.setdefault(name, {}).update({
                        "approach_speed_limited": bool(approach_limited),
                        "arm_velocity_limit_rad_s": limits.max_velocity,
                    })
                group_output, group_velocity = tracking_step(
                    {name: target[name]}, {name: target_velocity[name]},
                    {name: self._previous[name]}, {name: self._previous_velocity[name]},
                    dt_s=self._dt_s, position_gain=self._alpha / self._dt_s, limits=limits,
                )
                output.update(group_output)
                velocities.update(group_velocity)
        else:
            target = {
                name: self._previous[name] + self._alpha * (values[0] - self._previous[name])
                for name, values in reference.groups.items()
            }
            output, velocities = limit_step(
                target, self._previous, self._previous_velocity,
                dt_s=self._dt_s, limits=self._limits,
            )
        for name in reference.hold_groups:
            velocity = self._previous_velocity[name]
            velocities[name] = decelerate_velocity(
                velocity, self._limits.max_acceleration, self._dt_s
            )
            output[name] = self._previous[name] + velocities[name] * self._dt_s
        self._shape_grippers(now_ns, reference, output, velocities, state)
        self._previous = copy_group_vector(output)
        self._previous_velocity = copy_group_vector(velocities)
        return RobotCommand(groups=output, monotonic_ns=now_ns, plan_id=reference.plan_id)
