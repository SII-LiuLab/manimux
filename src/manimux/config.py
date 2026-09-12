from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    task: str
    output_dir: Path = Path("./data")
    max_steps: int = Field(default=500, gt=0)
    experiment_mode: bool = False
    layout_id: str = ""


class RobotConfig(StrictModel):
    driver: str
    config: Path | None = None
    control_hz: float = Field(default=100.0, gt=0)
    group_dims: dict[str, int]
    options: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_group_dims(self) -> RobotConfig:
        if not self.group_dims or any(dim <= 0 for dim in self.group_dims.values()):
            raise ValueError("robot group_dims must contain positive dimensions")
        return self


class SensorConfig(StrictModel):
    name: str
    driver: str
    width: int = Field(default=64, gt=0)
    height: int = Field(default=48, gt=0)
    fps: float = Field(default=30.0, gt=0)
    options: dict[str, object] = Field(default_factory=dict)


class ExpectedBackendConfig(StrictModel):
    server: str | None = None
    model: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> ExpectedBackendConfig:
        if self.server is None and not self.model:
            raise ValueError("policy.expected_backend must declare server or model identity")
        return self


class PolicyConfig(StrictModel):
    worker: str
    adapter: str
    device: str = "cpu"
    action_dt_s: float = Field(default=0.05, gt=0)
    trajectory_duration_s: float | None = Field(default=None, gt=0)
    timeout_s: float = Field(default=1.0, gt=0)
    horizon_steps: int = Field(default=20, gt=1)
    inference_delay_s: float = Field(default=0.04, ge=0)
    startup_timeout_s: float = Field(default=30.0, gt=0)
    action_decoding: Literal["inline", "process"] = "inline"
    expected_backend: ExpectedBackendConfig | None = None
    options: dict[str, object] = Field(default_factory=dict)

    @property
    def effective_action_dt_s(self) -> float:
        """Spacing between policy points, optionally derived from a total duration."""

        if self.trajectory_duration_s is None:
            return self.action_dt_s
        return self.trajectory_duration_s / (self.horizon_steps - 1)


class ExecutorLimitsConfig(StrictModel):
    max_velocity: float = Field(default=2.0, gt=0)
    max_acceleration: float = Field(default=8.0, gt=0)
    position_limit_abs: float = Field(default=3.14, gt=0)


class GripperHysteresisConfig(StrictModel):
    """Optional last-mile shaping for grippers embedded in joint groups."""

    mode: Literal["hysteresis", "continuous"] = "hysteresis"
    group_indices: dict[str, int]
    close_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    open_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_closed_s: float = Field(default=0.0, ge=0.0)
    open_confirm_s: float = Field(default=0.0, ge=0.0)
    max_velocity: float = Field(default=3.0, gt=0.0)
    max_acceleration: float = Field(default=12.0, gt=0.0)
    closed_value: float = Field(default=0.0, ge=0.0, le=1.0)
    open_value: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_hysteresis(self) -> GripperHysteresisConfig:
        if not self.group_indices:
            raise ValueError("gripper group_indices must not be empty")
        if any(not name or index < 0 for name, index in self.group_indices.items()):
            raise ValueError(
                "gripper group_indices must map non-empty names to non-negative indices"
            )
        if self.close_threshold >= self.open_threshold:
            raise ValueError("gripper close_threshold must be below open_threshold")
        if self.closed_value >= self.open_value:
            raise ValueError("gripper closed_value must be below open_value")
        return self


class GripperReleaseGuardConfig(StrictModel):
    mode: Literal["latched_release"] = "latched_release"
    kinematics: str = "yam"
    kinematics_options: dict[str, object] = Field(default_factory=dict)
    position_tolerance_m: float = Field(default=0.02, gt=0, le=0.05)
    phase_timeout_s: float = Field(default=2.0, gt=0, le=10.0)


class GripperGraspGuardConfig(StrictModel):
    """Finish closing at the captured pose before permitting a new arm target."""

    position_tolerance_m: float = Field(default=0.02, gt=0, le=0.05)
    rotation_tolerance_rad: float = Field(default=0.0872664626, gt=0, le=0.174532926)
    settle_s: float = Field(default=0.15, gt=0, le=1.0)
    aperture_stability: float = Field(default=0.01, gt=0, le=0.05)
    phase_timeout_s: float = Field(default=2.0, gt=0, le=10.0)
    approach_max_velocity: float | None = Field(default=None, gt=0)


class SmoothConfig(ExecutorLimitsConfig):
    cutoff_hz: float = Field(default=8.0, gt=0)
    tracking_mode: Literal["legacy", "braking"] = "legacy"
    gripper: GripperHysteresisConfig | None = None
    release_guard: GripperReleaseGuardConfig | None = None
    grasp_guard: GripperGraspGuardConfig | None = None

    @model_validator(mode="after")
    def validate_release_guard(self) -> SmoothConfig:
        if self.grasp_guard and self.release_guard is None:
            raise ValueError("grasp_guard requires release_guard for shared pose tracking")
        if self.release_guard and (self.gripper is None or self.gripper.mode != "continuous"):
            raise ValueError("release_guard requires a continuous gripper")
        if self.release_guard and self.tracking_mode != "braking":
            raise ValueError("latched_release requires braking tracking")
        return self


class MPCConfig(ExecutorLimitsConfig):
    horizon_steps: int = Field(default=15, gt=1)
    dynamics_a: float = Field(default=0.85, gt=0, lt=1)
    tracking_weight: float = Field(default=10.0, gt=0)
    command_delta_weight: float = Field(default=1.0, ge=0)


class CommandSafetyConfig(StrictModel):
    """Executor-independent per-joint command envelope for real hardware."""

    position_lower: dict[str, list[float]] = Field(default_factory=dict)
    position_upper: dict[str, list[float]] = Field(default_factory=dict)
    max_velocity: dict[str, list[float]] = Field(default_factory=dict)
    max_acceleration: dict[str, list[float]] = Field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.position_lower)

    @model_validator(mode="after")
    def validate_envelope(self) -> CommandSafetyConfig:
        mappings = (
            self.position_lower,
            self.position_upper,
            self.max_velocity,
            self.max_acceleration,
        )
        populated = [bool(values) for values in mappings]
        if not any(populated):
            return self
        if not all(populated):
            raise ValueError(
                "execution.command_safety requires position_lower, position_upper, "
                "max_velocity, and max_acceleration together"
            )
        groups = set(self.position_lower)
        if not groups or any(set(values) != groups for values in mappings[1:]):
            raise ValueError(
                "execution.command_safety mappings must contain the same groups"
            )
        for group in groups:
            lower = self.position_lower[group]
            upper = self.position_upper[group]
            velocity = self.max_velocity[group]
            acceleration = self.max_acceleration[group]
            dimensions = {len(lower), len(upper), len(velocity), len(acceleration)}
            if dimensions == {0} or len(dimensions) != 1:
                raise ValueError(
                    f"execution.command_safety group {group!r} vectors must share "
                    "one non-zero dimension"
                )
            values = lower + upper + velocity + acceleration
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"execution.command_safety group {group!r} must be finite"
                )
            if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
                raise ValueError(
                    f"execution.command_safety group {group!r} has invalid position bounds"
                )
            if any(value <= 0 for value in velocity + acceleration):
                raise ValueError(
                    f"execution.command_safety group {group!r} rate limits must be positive"
                )
        return self


class RtcConfig(StrictModel):
    """Physical Intelligence real-time chunking (arXiv:2506.07339).

    ``s_min`` is the smallest number of chunk steps to execute before starting
    the next inference; the effective execution horizon is ``max(s_min, d)``
    where ``d`` is the delay forecast. ``beta`` clips the guidance weight, which
    is unbounded at flow time 0; scale it with the sampler's step count using
    ``(t^2+(1-t)^2)/(t(1-t))`` at ``t = 1/steps`` (5 steps -> ~4.25, 10 -> ~9.1).
    """

    min_execute_steps: int | None = Field(default=None, gt=0)
    initial_delay_steps: int = Field(default=4, ge=0)
    delay_buffer_size: int = Field(default=10, gt=0)
    beta: float = Field(default=5.0, gt=0)
    # RTC changes only *when* to infer and *what* to condition on. How a chunk
    # is executed -- timeline, smoothing, limits -- stays the default runtime's,
    # configured by ``execution.executor`` and ``execution.smooth`` as usual.


class TemporalEnsembleConfig(StrictModel):
    """ACT temporal ensembling with an asynchronous query cadence."""

    coefficient: float = Field(default=0.01, ge=0)
    query_interval_steps: int = Field(default=1, gt=0)


class AacConfig(StrictModel):
    """Adaptive Action Chunking sampling and selection parameters."""

    num_samples: int = Field(default=20, gt=1)
    motion_threshold: float = Field(default=3.0, ge=0)
    ee_stats_path: str | None = None
    chunk_id_selector: Literal["0", "mean", "backward"] = "0"
    backward_beta: float = Field(default=0.99, gt=0, le=1)


class PaintConfig(StrictModel):
    """PAINT asynchronous execution parameters (arXiv:2606.19774)."""

    execution_steps: int = Field(default=10, gt=0)
    initial_delay_steps: int = Field(default=4, gt=0)
    delay_buffer_size: int = Field(default=10, gt=0)


class DvacConfig(StrictModel):
    """Denoising-Variance Adaptive Chunking (arXiv:2606.03847v1)."""

    tail_steps: int = Field(default=5, gt=1)
    alpha: float = Field(default=2.0, ge=0)
    rolling_window_size: int = Field(default=5, gt=0)
    min_execution_steps: int = Field(default=1, gt=0)
    max_execution_steps: int | None = Field(default=None, gt=0)


class ExecutionConfig(StrictModel):
    runtime: str = Field(default="manimux", min_length=1)
    # Common tuning entrypoint, in policy action steps (not control ticks).
    # Each strategy retains its timing/capping semantics; see docs/chunk-steps.md.
    chunk_steps: int | None = Field(default=None, gt=0, strict=True)
    executor: Literal["direct", "smooth", "mpc"] = "smooth"
    inference_schedule: Literal["deadline", "single_inflight", "serial"] = "deadline"
    refill_threshold_s: float = Field(default=0.4, gt=0)
    commit_lead_s: float = Field(default=0.02, ge=0)
    max_plan_age_s: float = Field(default=1.0, gt=0)
    blend_steps: int = Field(default=2, ge=0)
    # Cap original source rows, before latency trimming; policy inference stays unchanged.
    max_chunk_steps: int | None = Field(default=None, ge=2)
    independent_group_decoding: bool = False
    decode_budget_ms: float = Field(default=40.0, gt=0, le=200)
    smooth: SmoothConfig = SmoothConfig()
    mpc: MPCConfig = MPCConfig()
    command_safety: CommandSafetyConfig = CommandSafetyConfig()
    rtc: RtcConfig = RtcConfig()
    temporal_ensemble: TemporalEnsembleConfig = TemporalEnsembleConfig()
    aac: AacConfig = AacConfig()
    paint: PaintConfig = PaintConfig()
    dvac: DvacConfig = DvacConfig()

    @model_validator(mode="before")
    @classmethod
    def resolve_chunk_steps(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("chunk_steps") is None:
            return value
        paths = {
            "manimux": (None, "max_chunk_steps"),
            "rtc": ("rtc", "min_execute_steps"),
            "paint": ("paint", "execution_steps"),
            "act_temporal_ensemble": ("temporal_ensemble", "query_interval_steps"),
            "dvac": ("dvac", "max_execution_steps"),
        }
        runtime = value.get("runtime", "manimux")
        if runtime not in paths:
            raise ValueError(
                f"execution.chunk_steps is not supported by runtime {runtime!r}; "
                "do not override an adaptive model-selected horizon with a fixed cadence"
            )
        section, field = paths[runtime]
        resolved = dict(value)
        if section is None:
            destination = resolved
        else:
            nested = value.get(section, {})
            if isinstance(nested, BaseModel):
                nested = nested.model_dump(exclude_unset=True)
            if not isinstance(nested, dict):
                raise ValueError(f"execution.{section} must be a mapping")
            destination = dict(nested)
            resolved[section] = destination
        steps = value["chunk_steps"]
        if destination.get(field) is not None and destination[field] != steps:
            path = f"{section}.{field}" if section else field
            raise ValueError(f"execution.chunk_steps conflicts with execution.{path}")
        destination[field] = steps
        return resolved

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> ExecutionConfig:
        if self.inference_schedule == "serial":
            if self.runtime != "manimux":
                raise ValueError("serial scheduling requires execution.runtime=manimux")
            if "refill_threshold_s" in self.model_fields_set:
                raise ValueError("serial scheduling does not use refill_threshold_s")
        if self.independent_group_decoding and (
            self.runtime != "manimux" or self.executor != "smooth"
            or self.smooth.tracking_mode != "braking"
            or self.smooth.gripper is None
            or self.smooth.gripper.mode != "continuous"
        ):
            raise ValueError(
                "independent group decoding requires braking smooth with continuous grippers"
            )
        if self.max_chunk_steps is not None and (
            self.runtime != "manimux" or self.executor not in {"smooth", "direct", "mpc"}
        ):
            raise ValueError("max_chunk_steps requires the ordinary ManiMux joint timeline")
        if self.runtime == "rtc":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by RTC: {names}")
        if self.runtime == "act_temporal_ensemble":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(
                    f"execution fields are not used by ACT temporal ensembling: {names}"
                )
            if self.blend_steps != 0:
                raise ValueError(
                    "ACT temporal ensembling requires execution.blend_steps=0 "
                    "to avoid modifying the ensembled trajectory"
                )
        if self.runtime == "aac":
            if not self.aac.ee_stats_path:
                raise ValueError("execution.aac.ee_stats_path is required by AAC")
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by AAC: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "AAC requires execution.blend_steps=0 so the selected chunk "
                    "is not rewritten at commit"
                )
        if self.runtime == "paint":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by PAINT: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "PAINT requires execution.blend_steps=0 so A[s:s+d] is not "
                    "rewritten before it becomes the next prefix condition"
                )
        if self.runtime == "autohorizon":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by AutoHorizon: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "AutoHorizon requires execution.blend_steps=0 so the selected "
                    "official execution horizon is not rewritten at commit"
                )
        if self.runtime == "dvac":
            ignored = {"inference_schedule", "refill_threshold_s"}.intersection(
                self.model_fields_set
            )
            if ignored:
                names = ", ".join(sorted(ignored))
                raise ValueError(f"execution fields are not used by DVAC: {names}")
            if self.blend_steps != 0:
                raise ValueError(
                    "DVAC requires execution.blend_steps=0 so the selected "
                    "paper execution horizon is not rewritten at commit"
                )
        return self


class ViewerConfig(StrictModel):
    enabled: bool = False
    robot_adapter: str = ""
    policy_label: str = ""
    camera_hz: float = Field(default=5.0, ge=0)


class RecordingConfig(StrictModel):
    enabled: Literal[True] = True
    video_fps: float = Field(default=0.0, ge=0.0)
    video_codec: str = Field(default="mp4v", min_length=4, max_length=4)
    video_queue_size: int = Field(default=8, gt=0)


class ManiMuxConfig(StrictModel):
    run: RunConfig
    robot: RobotConfig
    sensors: list[SensorConfig] = []
    policy: PolicyConfig
    execution: ExecutionConfig = ExecutionConfig()
    viewer: ViewerConfig = ViewerConfig()
    recording: RecordingConfig = RecordingConfig()

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> ManiMuxConfig:
        if (self.execution.inference_schedule == "serial"
                and self.policy.action_decoding != "inline"):
            raise ValueError(
                "serial scheduling requires inline action decoding without latency trimming"
            )
        if (self.execution.chunk_steps is not None
                and self.execution.chunk_steps > self.policy.horizon_steps):
            raise ValueError("execution.chunk_steps must not exceed policy.horizon_steps")
        if self.execution.independent_group_decoding and self.policy.action_decoding != "process":
            raise ValueError("independent group decoding requires process action decoding")
        if (
            self.execution.max_chunk_steps is not None
            and self.execution.max_chunk_steps > self.policy.horizon_steps
        ):
            raise ValueError("max_chunk_steps must not exceed policy.horizon_steps")
        command_safety = self.execution.command_safety
        if command_safety.configured:
            expected_groups = set(self.robot.group_dims)
            if set(command_safety.position_lower) != expected_groups:
                raise ValueError(
                    "execution.command_safety groups must exactly match robot.group_dims"
                )
            for group, dimension in self.robot.group_dims.items():
                if len(command_safety.position_lower[group]) != dimension:
                    raise ValueError(
                        f"execution.command_safety group {group!r} must have "
                        f"{dimension} values"
                    )
        if self.execution.runtime == "rtc":
            delay = self.execution.rtc.initial_delay_steps
            if 2 * delay > self.policy.horizon_steps:
                raise ValueError("RTC requires 2 * initial_delay_steps <= policy.horizon_steps")
        if self.execution.runtime == "act_temporal_ensemble":
            query_interval = self.execution.temporal_ensemble.query_interval_steps
            if query_interval >= self.policy.horizon_steps:
                raise ValueError(
                    "ACT temporal ensembling requires query_interval_steps "
                    "< policy.horizon_steps so consecutive chunks overlap"
                )
        if self.execution.runtime == "paint":
            execution = self.execution.paint.execution_steps
            delay = self.execution.paint.initial_delay_steps
            horizon = self.policy.horizon_steps
            if not delay <= execution <= horizon - delay:
                raise ValueError(
                    "PAINT requires initial_delay_steps <= execution_steps "
                    "<= horizon_steps - initial_delay_steps"
                )
        if self.execution.runtime == "dvac":
            dvac = self.execution.dvac
            maximum = dvac.max_execution_steps or self.policy.horizon_steps
            if not dvac.min_execution_steps <= maximum <= self.policy.horizon_steps:
                raise ValueError(
                    "DVAC requires min_execution_steps <= max_execution_steps "
                    "<= policy.horizon_steps"
                )
        return self


def load_config(path: str | Path) -> ManiMuxConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ManiMuxConfig.model_validate(raw)
