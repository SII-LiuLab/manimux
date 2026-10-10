"""Selected-prefix execution with serial defaults and optional async scheduling."""

from collections.abc import Mapping

from manimux.runtime.inference import CommitSettings, InferenceSubmission
from manimux.runtime.safety import RuntimeState
from manimux.types import ActionChunk, InferenceRequest, copy_group_vector


class SynchronousChunkStrategy:
    """Shared request construction and metadata-selected execution prefixes.

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

    @property
    def discard_plans_while_paused(self) -> bool:
        return self._config["inference"]["inference_schedule"] != "serial"

    def reset(self) -> None:
        pass

    def request_options(self) -> dict:
        return {}

    def submission_fields(self) -> dict:
        return {}

    def build_warmup_submission(
        self, *, session_id, request_seq, now_ns, snapshot, adapter, conditioned=False,
    ):
        settings = self._config["inference"]
        legacy = settings["inference_schedule"] == "serial" and not (
            settings.get("temporal_ensemble") or {}
        ).get("enabled", False)
        return self._make_submission(
            session_id=session_id, request_seq=request_seq, now_ns=now_ns,
            snapshot=snapshot, adapter=adapter, default_sampling=legacy,
        )

    def _make_submission(
        self, *, session_id, request_seq, now_ns, snapshot, adapter, default_sampling=False,
    ):
        # Preserve legacy serial warmup; new combinations exercise their own sampler.
        request_type = InferenceRequest if default_sampling else self.request_type
        return InferenceSubmission(
            request=request_type(
                session_id=session_id, request_seq=request_seq,
                observation_time_ns=snapshot.state.monotonic_ns,
                deadline_ns=now_ns + int(self._config["policy"]["timeout_s"] * 1e9),
                observation=adapter.build_observation(snapshot),
                instruction=self._config["run"]["task"],
                **({} if default_sampling else self.request_options()),
            ),
            event_fields={"conditioned": False} if default_sampling else self.submission_fields(),
        )

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
        # Serial keeps its original cadence; asynchronous admission is scheduler-owned.
        if (
            runtime_state != RuntimeState.RUNNING
            or (
                self._config["inference"]["inference_schedule"] == "serial"
                and timeline.remaining_ns(now_ns) > 0
            )
        ):
            return None
        return self._make_submission(
            session_id=session_id, request_seq=request_seq, now_ns=now_ns,
            snapshot=snapshot, adapter=adapter,
        )

    def commit_settings(self, *, response, measured, last_command) -> CommitSettings:
        # The config owns seam blending; synchronous strategies anchor it at measured state.
        return CommitSettings(
            current_command=copy_group_vector(measured),
            blend_steps=self._config["inference"]["blend_policy_steps"],
            anchor_source="measured_state",
        )

    def decode_handoff(self, *, response) -> bool:
        """response: model output about to be decoded; this strategy keeps its own trajectory."""
        del response
        return False

    def prepare_chunk(self, *, chunk, response, now_ns) -> ActionChunk:
        del now_ns
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
        # Select the execution prefix without changing the real observation timestamp.
        return ActionChunk(
            plan_id=chunk.plan_id,
            request_seq=chunk.request_seq,
            observation_time_ns=chunk.observation_time_ns,
            created_time_ns=chunk.created_time_ns,
            action_space=chunk.action_space,
            dt_ns=chunk.dt_ns,
            groups={name: values[:execution_steps].copy() for name, values in chunk.groups.items()},
            source_offset_steps=chunk.source_offset_steps,
            metadata=dict(chunk.metadata),
            hold_from_step={
                name: step for name, step in chunk.hold_from_step.items() if step < execution_steps
            },
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
