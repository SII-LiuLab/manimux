"""Experimental RTC with shared path retiming and bounded progress handoff.

The model still samples H actions on its ordinary playback grid. Only accepted
execution trajectories are retimed; subsequent conditions sample that executable
trajectory, including its bridge, on the next observation's fixed H-row grid.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from manimux.policies.base import action_interval_ns
from manimux.runtime.inference import InferenceSubmission
from manimux.runtime.rtc.mask import soft_mask
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.rtc.strategy import RtcInferenceStrategy
from manimux.runtime.safety import RuntimeState
from manimux.types import RuntimeTrajectory, copy_group_vector

DEFAULTS = {
    "strength": 0.5,
    "rotation_radius_m": 0.10,
    "stationary_distance_m": 0.0002,
    "gripper_event_delta": 0.08,
    "gripper_motion_delta": 0.005,
    "min_interval_ratio": 0.35,
    "match_window_s": 0.6,
    "match_resolution_steps": 0.25,
    "max_position_gap_m": 0.025,
    "max_rotation_gap_rad": 0.30,
    "max_gripper_gap": 0.15,
    "min_direction_cosine": 0.0,
    "bridge_s": 0.20,
    "min_future_s": 0.6,
    "runtime_substeps": 8,
}


def retimed_parameters(options):
    unknown = set(options) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown rtc_retime settings: {sorted(unknown)}")
    values = {**DEFAULTS, **options}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"rtc_retime.{key} must be finite numeric")
    for key in DEFAULTS.keys() - {"strength", "min_direction_cosine"}:
        if values[key] <= 0:
            raise ValueError(f"rtc_retime.{key} must be positive")
    if not 0 <= values["strength"] <= 1 or not 0 < values["min_interval_ratio"] <= 1:
        raise ValueError("rtc_retime strength/interval ratio must lie in [0,1]/(0,1]")
    if not -1 <= values["min_direction_cosine"] <= 1:
        raise ValueError("rtc_retime.min_direction_cosine must lie in [-1,1]")
    if type(values["runtime_substeps"]) is not int or values["runtime_substeps"] < 2:
        raise ValueError("rtc_retime.runtime_substeps must be an integer >= 2")
    return values


def retime_knots(distance, protected, dt_s, options):
    """Allocate the same total playback duration across synchronized intervals."""
    distance = np.asarray(distance, dtype=float)
    protected = np.asarray(protected, dtype=bool)
    durations = np.full(len(distance), dt_s)
    moving = ~protected
    if moving.any() and distance[moving].sum() > 1e-12:
        uniform = distance[moving] / distance[moving].mean()
        ratio = (1 - options["strength"]) + options["strength"] * uniform
        # A positive floor avoids deleting low-motion portions; preserve the budget.
        floor = options["min_interval_ratio"]
        excess = np.maximum(ratio - floor, 0)
        ratio = (
            floor + excess * (len(excess) * (1 - floor) / excess.sum())
            if excess.sum()
            else np.ones_like(excess)
        )
        durations[moving] *= ratio
    return np.r_[0.0, np.cumsum(durations)]


def interpolate_rows(rows, positions):
    p = np.clip(np.asarray(positions), 0, len(rows) - 1)
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, len(rows) - 1)
    return rows[lo] * (1 - (p - lo))[..., None] + rows[hi] * (p - lo)[..., None]


def rotation_distance(a, b):
    return np.arccos(np.clip((np.einsum("...ij,...ij->...", a, b) - 1) / 2, -1, 1))


class RetimedRtcStrategy(RtcInferenceStrategy):
    """Fixed model grid, variable source progress, executable-reference conditions."""

    def __init__(self, config):
        self.options = retimed_parameters(config["inference"].get("rtc_retime", {}))
        inf = config["inference"]
        if (
            inf["algorithm"] != "rtc"
            or inf["inference_schedule"] != "single_inflight"
            or inf["action_start_mode"] != "drop_infer_latency"
            or inf["handoff"] != "blend"
            or inf["handoff_skip_steps"]
            or inf["blend_policy_steps"]
            or inf.get("max_chunk_policy_steps") is not None
            or inf.get("independent_group_decoding", False)
            or inf["rtc"].get("condition_source") == "executor"
            or (inf.get("temporal_ensemble") or {}).get("enabled", False)
            or config.get("execution", {}).get("ik", {}).get("mode", "chunk") != "chunk"
        ):
            raise ValueError(
                "retimed RTC requires single-inflight joint RTC, drop_infer_latency, "
                "no extra skip/blend/cap/ensemble or independent group decoding"
            )
        from manimux.embodiments.robot import RobotModel

        robot = RobotModel.from_config(config["robot"]["config"])
        self.models = robot.kinematics.models
        self.arm_dofs = {
            name: config["robot"]["group_layouts"][name]["arm_dofs"]
            for name in config["robot"]["group_dims"]
        }
        for name, dims in config["robot"]["group_dims"].items():
            if name not in self.models or dims != self.arm_dofs[name] + 1:
                raise ValueError(
                    "retimed RTC requires an FK arm and one normalized gripper per group"
                )
        super().__init__(config)

    @property
    def name(self):
        return "rtc_retime"

    def reset(self):
        super().reset()
        self._observed = {}

    def _forget_request(self, request_seq):
        super()._forget_request(request_seq)
        self._observed.pop(request_seq, None)

    def build_submission(
        self,
        *,
        session_id,
        request_seq,
        now_ns,
        snapshot,
        adapter,
        timeline,
        request_state,
        runtime_state,
    ):
        if runtime_state != RuntimeState.RUNNING:
            return None
        dt = action_interval_ns(self._config["policy"])
        horizon = self._config["policy"]["horizon_policy_steps"]
        delay = max(self._delay_forecast)
        executed = 0
        condition = weights = None
        forecast = 0
        if self._active_rows is not None:
            executed = self._active_offset + timeline.cursor(now_ns)
            # Both progress and remaining budget refer to the accepted execution clock.
            trigger = self.execution_horizon(len(self._active_rows), delay)
            if executed < trigger and timeline.remaining_ns(now_ns) > (delay + 1) * dt:
                return None
            samples = [
                timeline.sample_runtime(snapshot.state.monotonic_ns + i * dt)
                for i in range(horizon)
            ]
            valid = np.array([s is not None for s in samples])
            # A valid prefix only; never condition on extrapolated or resurrected plans.
            overlap = int(np.flatnonzero(~valid)[0]) if not valid.all() else horizon
            overlap = min(overlap, horizon - delay)
            if delay <= overlap and delay <= horizon - overlap and overlap > 0:
                condition = np.zeros((horizon, sum(self._config["robot"]["group_dims"].values())))
                for i in range(overlap):
                    condition[i] = np.concatenate([samples[i][g] for g in self._group_order])
                weights = soft_mask(horizon, horizon - overlap, delay)
                forecast = delay
        request = RtcInferenceRequest(
            session_id=session_id,
            request_seq=request_seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=now_ns + int(self._config["policy"]["timeout_s"] * 1e9),
            observation=adapter.build_observation(snapshot),
            instruction=self._config["run"]["task"],
            action_condition=condition,
            condition_weights=weights,
            action_condition_space=timeline.action_space,
            rtc_beta=self._rtc_beta,
            condition_time_ns=snapshot.state.monotonic_ns,
        )
        if condition is not None:
            self._conditioned_requests.add(request_seq)
        self._request_started_ns[request_seq] = now_ns
        self._request_observation_ns[request_seq] = snapshot.state.monotonic_ns
        self._request_forecast[request_seq] = forecast
        self._observed[request_seq] = copy_group_vector(snapshot.state.groups)
        return InferenceSubmission(
            request,
            {
                "conditioned": condition is not None,
                "forecast_delay": forecast,
                "executed_steps": executed,
                "condition_source": "retimed_runtime",
                "condition_handoff_skip_steps": 0,
            },
        )

    def prepare_chunk(self, *, chunk, response, now_ns):
        super().prepare_chunk(chunk=chunk, response=response, now_ns=now_ns)
        if chunk.action_space != "joint_position" or chunk.source_offset_steps or chunk.handoff:
            raise ValueError(
                "retimed RTC requires complete decoded joint predictions without waypoint handoff"
            )
        if chunk.dt_ns != action_interval_ns(self._config["policy"]):
            raise ValueError("retimed RTC playback grid mismatch")
        return chunk

    def _geometry(self, groups):
        return {
            name: np.array([self.models[name].fk(q) for q in rows]) for name, rows in groups.items()
        }

    def prepare_timed_handoff(
        self, *, chunk, now_ns, timeline, measured, current_command, command_velocity
    ):
        opt = self.options
        dt = chunk.dt_ns / 1e9
        h = chunk.horizon_steps
        geo = self._geometry(chunk.groups)
        increments = []
        protected = np.zeros(h - 1, dtype=bool)
        gripper_events = np.zeros(h - 1, dtype=bool)
        arm_lengths = {}
        for name, transforms in geo.items():
            xyz = transforms[:, :3, 3]
            translation = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            rotation = rotation_distance(transforms[:-1, :3, :3], transforms[1:, :3, :3])
            increments.append(np.hypot(translation, opt["rotation_radius_m"] * rotation))
            arm_lengths[name] = np.r_[0, np.cumsum(translation)]
            gripper_events |= (
                np.abs(np.diff(chunk.groups[name][:, -1])) >= opt["gripper_event_delta"]
            )
            protected |= np.abs(np.diff(chunk.groups[name][:, -1])) >= opt["gripper_motion_delta"]
        protected |= gripper_events
        distance = np.max(increments, axis=0)
        protected |= distance < opt["stationary_distance_m"]
        # Keep the hard RTC prefix's original time allocation; it represents overlap.
        prefix = self._request_forecast.get(chunk.request_seq, 0)
        protected[:prefix] = True
        knots = retime_knots(distance, protected, dt, opt)
        elapsed = max(0, (now_ns - chunk.observation_time_ns) / 1e9)
        phase = float(np.interp(elapsed, knots, np.arange(h)))
        matched = False
        match_info = {}
        old = timeline.sample_runtime(now_ns)
        if old is not None:
            # Match physical progress since observation, with outgoing-reference arc
            # length as a curvature-aware prior and current measured pose as a gate.
            history_times = np.linspace(chunk.observation_time_ns, now_ns, 17).astype(np.int64)
            history = [timeline.sample_runtime(int(t)) for t in history_times]
            observed = self._observed.get(chunk.request_seq, measured)
            current_geo = self._geometry({n: np.asarray([q]) for n, q in measured.items()})
            command_geo = self._geometry({n: np.asarray([q]) for n, q in current_command.items()})
            old_distance, directions = {}, {}
            for name in self._group_order:
                points = [self.models[name].fk(s[name])[:3, 3] for s in history if s is not None]
                displacement = (
                    current_geo[name][0, :3, 3] - self.models[name].fk(observed[name])[:3, 3]
                )
                old_distance[name] = (
                    float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                    if len(points) >= 2
                    else float(np.linalg.norm(displacement))
                )
                before = timeline.sample_runtime(now_ns - 50_000_000)
                directions[name] = (
                    (
                        self.models[name].fk(old[name])[:3, 3]
                        - self.models[name].fk(before[name])[:3, 3]
                    )
                    if before
                    else displacement
                )
            upper = float(
                np.interp(
                    min(knots[-1] - opt["min_future_s"], elapsed + opt["match_window_s"]),
                    knots,
                    np.arange(h),
                )
            )
            # Do not skip a gripper event; a shared phase preserves dual-arm ordering.
            events = np.flatnonzero(gripper_events)
            if len(events):
                upper = min(upper, float(events[0]))
            for rows in chunk.groups.values():
                # A gradual close/open is also an event, even if no single step
                # crosses the event threshold. Do not skip its cumulative travel.
                aperture_travel = np.r_[0, np.cumsum(np.abs(np.diff(rows[:, -1])))]
                crossings = np.flatnonzero(aperture_travel >= opt["gripper_event_delta"])
                if len(crossings):
                    upper = min(upper, max(0, float(crossings[0] - 1)))
            candidates = np.arange(0, max(0, upper) + 1e-9, opt["match_resolution_steps"])
            candidates = np.unique(np.r_[candidates, min(phase, max(0, upper))])
            scores = np.zeros(len(candidates))
            compatible = np.ones(len(candidates), dtype=bool)
            for name in self._group_order:
                q = interpolate_rows(chunk.groups[name], candidates)
                xyz = interpolate_rows(geo[name][:, :3, 3], candidates)
                rotations = Slerp(np.arange(h), Rotation.from_matrix(geo[name][:, :3, :3]))(
                    candidates
                ).as_matrix()
                gap = np.linalg.norm(xyz - current_geo[name][0, :3, 3], axis=1)
                command_gap = np.linalg.norm(xyz - command_geo[name][0, :3, 3], axis=1)
                angle = rotation_distance(rotations, current_geo[name][0, :3, :3])
                grip = np.abs(q[:, -1] - measured[name][-1])
                # Rank fractional candidates from source poses; verify the winner
                # with exact FK below, keeping search out of the control budget.
                lo = np.minimum(candidates.astype(int), h - 2)
                tangent = geo[name][lo + 1, :3, 3] - geo[name][lo, :3, 3]
                denom = np.linalg.norm(tangent, axis=1) * np.linalg.norm(directions[name])
                cosine = np.ones(len(candidates))
                np.divide(tangent @ directions[name], denom, out=cosine, where=denom > 1e-9)
                compatible &= (
                    (np.maximum(gap, command_gap) <= opt["max_position_gap_m"])
                    & (angle <= opt["max_rotation_gap_rad"])
                    & (grip <= opt["max_gripper_gap"])
                    & (cosine >= opt["min_direction_cosine"])
                )
                progress_error = (
                    np.interp(candidates, np.arange(h), arm_lengths[name]) - old_distance[name]
                )
                scores += (gap / opt["max_position_gap_m"]) ** 2 + (
                    progress_error / opt["max_position_gap_m"]
                ) ** 2
                scores += (angle / opt["max_rotation_gap_rad"]) ** 2 + (
                    grip / opt["max_gripper_gap"]
                ) ** 2
                scores += 0.25 * (1 - cosine)
            if not compatible.any():
                raise ValueError("retimed_handoff_no_compatible_progress")
            scores[~compatible] = np.inf
            best_index = None
            for index in np.argsort(scores):
                if not np.isfinite(scores[index]):
                    break
                valid = True
                for name in self._group_order:
                    pose = self.models[name].fk(
                        interpolate_rows(chunk.groups[name], candidates[index])
                    )
                    if (
                        np.linalg.norm(pose[:3, 3] - current_geo[name][0, :3, 3])
                        > opt["max_position_gap_m"]
                        or np.linalg.norm(pose[:3, 3] - command_geo[name][0, :3, 3])
                        > opt["max_position_gap_m"]
                        or rotation_distance(pose[:3, :3], current_geo[name][0, :3, :3])
                        > opt["max_rotation_gap_rad"]
                    ):
                        valid = False
                        break
                if valid:
                    best_index = int(index)
                    break
            if best_index is None:
                raise ValueError("retimed_handoff_no_compatible_progress")
            phase = float(candidates[best_index])
            matched = True
            match_info = {
                "overlap_reference_distance_m": old_distance,
                "match_score": float(scores[best_index]),
            }
        phase_time = float(np.interp(phase, np.arange(h), knots))
        remaining = knots[-1] - phase_time
        if remaining < max(opt["min_future_s"], opt["bridge_s"] + dt):
            raise ValueError("retimed_handoff_insufficient_future")
        # Round the surviving endpoint up to the model playback grid. No second
        # skip is applied. Full-path duration is preserved, removed progress is not
        # stretched back to a full H-step chunk.
        count = int(np.ceil((elapsed + remaining) / dt)) + 1
        end_ns = chunk.observation_time_ns + (count - 1) * chunk.dt_ns
        runtime_dt = max(1, chunk.dt_ns // opt["runtime_substeps"])
        dense_count = (end_ns - now_ns) // runtime_dt + 1
        dense_times = np.arange(dense_count) * runtime_dt / 1e9
        available = (end_ns - now_ns) / 1e9
        source_phase = np.interp(
            phase_time + dense_times * remaining / available, knots, np.arange(h)
        )
        dense = {}
        for name, rows in chunk.groups.items():
            if chunk.runtime_trajectory is None:
                dense[name] = interpolate_rows(rows, source_phase)
            else:
                original = chunk.runtime_trajectory
                positions = (
                    (chunk.observation_time_ns - original.start_time_ns)
                    + source_phase * chunk.dt_ns
                ) / original.dt_ns
                dense[name] = interpolate_rows(original.groups[name], positions)
        # Fade a position/velocity correction to zero, preserving the retimed path
        # beyond the bridge. Grippers use a monotone weight, never cubic velocity.
        bridge = opt["bridge_s"]
        u = np.clip(dense_times / bridge, 0, 1)
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        for name, rows in dense.items():
            q0 = rows[0].copy()
            end_grip = float(np.interp(bridge, dense_times, rows[:, -1]))
            v0 = (rows[1] - rows[0]) / (runtime_dt / 1e9)
            rows += h00[:, None] * (current_command[name] - q0)
            velocity_error = command_velocity[name] - v0
            velocity_error[-1] = 0
            rows += h10[:, None] * bridge * velocity_error
            inside = dense_times < bridge
            weight = u[inside] ** 2 * (3 - 2 * u[inside])
            rows[inside, -1] = (1 - weight) * current_command[name][-1] + weight * end_grip
            rows[:, -1] = np.clip(rows[:, -1], 0, 1)
        safety = self._config["executor"]["command_safety"]
        for name, rows in dense.items():
            lower = (safety.get("position_lower") or {}).get(name)
            upper = (safety.get("position_upper") or {}).get(name)
            if (
                not np.isfinite(rows).all()
                or (lower is not None and np.any(rows < lower))
                or (upper is not None and np.any(rows > upper))
            ):
                raise ValueError("retimed_bridge_outside_joint_bounds")
        # Canonical execution rows mirror the dense executable path, rather than
        # retaining stale pre-retiming rows for the next RTC request.
        times = chunk.observation_time_ns + np.arange(count, dtype=np.int64) * chunk.dt_ns
        groups = {
            name: interpolate_rows(rows, (times - now_ns) / runtime_dt)
            for name, rows in dense.items()
        }
        mapping = np.interp(
            phase_time + np.maximum(0, (times - now_ns) / 1e9) * remaining / available,
            knots,
            np.arange(h),
        )
        metadata = {
            **chunk.metadata,
            "rtc_retime": {
                "schema": "rtc-retime-v1",
                "strength": opt["strength"],
                "native_speed_ratio": float(self._config["policy"]["action_dt_s"] / dt),
                "source_knots_s": knots.tolist(),
                "source_phase_at_execution_rows": mapping.tolist(),
                "matched_source_phase": phase,
                "matched": matched,
                "protected_prefix_steps": prefix,
                "original_horizon": h,
                "execution_horizon": count,
                "bridge_s": bridge,
                "handoff_time_ns": now_ns,
                "remaining_s": available,
                **match_info,
            },
        }
        return replace(
            chunk,
            groups=groups,
            runtime_trajectory=RuntimeTrajectory(now_ns, runtime_dt, dense),
            metadata=metadata,
        )

    def on_plan_accepted(self, *, chunk, result, response, now_ns):
        events = super().on_plan_accepted(
            chunk=chunk, result=result, response=response, now_ns=now_ns
        )
        self._observed.pop(response.request_seq, None)
        return {**events, "rtc_condition_source": "retimed_runtime"}
