"""Shared cadence for policies that finish one selected chunk before requesting another."""

from collections.abc import Mapping

from manimux.runtime.inference import CommitSettings, InferenceSubmission
from manimux.runtime.safety import RuntimeState
from manimux.types import ActionChunk, InferenceRequest, copy_group_vector


class SynchronousChunkStrategy:
    """One in-flight request; each returned chunk starts at the measured state.

    Subclasses declare their request type and sampling options. A strategy with
    horizon_metadata also uses the server's execution_steps to select a prefix.
    Model-specific selection algorithms remain in the model server.
    """

    name: str
    request_type: type[InferenceRequest]
    horizon_metadata: str | None = None
    label: str

    def __init__(self, config: dict) -> None:
        self._config = config

    @property
    def control_mode(self) -> str:
        return self.name

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({self.name})

    def reset(self) -> None:
        pass

    def request_options(self) -> dict:
        return {}

    def submission_fields(self) -> dict:
        return {}

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
    ) -> InferenceSubmission | None:
        # 当前动作执行完且无在途请求时，才采集下一次推理的观测。
        if (
            request_state.in_flight
            or runtime_state != RuntimeState.RUNNING
            or timeline.remaining_ns(now_ns) > 0
        ):
            return None
        return InferenceSubmission(
            request=self.request_type(
                session_id=session_id,
                request_seq=request_seq,
                observation_time_ns=snapshot.state.monotonic_ns,
                deadline_ns=now_ns + int(self._config["policy"]["timeout_s"] * 1_000_000_000),
                observation=adapter.build_observation(snapshot),
                instruction=self._config["run"]["task"],
                **self.request_options(),
            ),
            event_fields=self.submission_fields(),
        )

    def commit_settings(self, *, response, measured, last_command) -> CommitSettings:
        # 保持原同步策略：从实测状态开始，不混合上一块的预测轨迹。
        return CommitSettings(
            current_command=copy_group_vector(measured),
            blend_steps=0,
            anchor_source="measured_state",
        )

    def prepare_chunk(self, *, chunk, response, now_ns) -> ActionChunk:
        execution_steps = chunk.horizon_steps
        if self.horizon_metadata is not None:
            raw = response.raw_action
            metadata = raw.get(self.horizon_metadata) if isinstance(raw, Mapping) else None
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{self.label} response is missing metadata")
            execution_steps = metadata.get("execution_steps")
            if (
                not isinstance(execution_steps, int)
                or isinstance(execution_steps, bool)
                or not 1 <= execution_steps <= chunk.horizon_steps
            ):
                raise ValueError(
                    f"{self.label} execution_steps must satisfy "
                    f"1 <= e <= {chunk.horizon_steps}, got {execution_steps!r}"
                )
        # 保留原行为：在收到动作时重设起始时间，只复制选中的执行段。
        return ActionChunk(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            observation_time_ns=now_ns,
            created_time_ns=chunk.created_time_ns,
            action_space=chunk.action_space,
            dt_ns=chunk.dt_ns,
            groups={name: values[:execution_steps].copy() for name, values in chunk.groups.items()},
        )

    def on_plan_accepted(self, *, chunk, result, response, now_ns) -> dict[str, object]:
        return {
            "selected_horizon_steps": chunk.horizon_steps,
            "trimmed_steps": result.trimmed_steps,
        }

    def on_response_rejected(self, response) -> None:
        pass

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        return []

    def on_tick(self, *, steps: int, loop_ms: float, control_dt_ns: int) -> None:
        pass
