"""Runtime-owned ChunkTrust selection and original-policy-step history.

The source selector sees raw model velocity and consistently decoded joint targets.
Only plan acceptance commits its posterior. Command feedback admits source rows
to history; their values come from the model prediction before interpolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

from manimux.policies.base import action_interval, attach_sampling_stats
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.chunktrust.config import ChunkTrustOptions
from manimux.runtime.chunktrust.core import ChunkTrustAdapter, PreparedDecision
from manimux.runtime.chunktrust.trace import (
    TraceProfile,
    extract_velocity,
    validate_capabilities,
)
from manimux.runtime.feedback import CommandFeedback
from manimux.runtime.inference import InferenceSubmission
from manimux.runtime.synchronous import SynchronousChunkStrategy
from manimux.runtime.timeline import CommitResult
from manimux.types import (
    ActionChunk,
    ActionContext,
    InferenceRequest,
    InferenceResponse,
    copy_action_chunk,
)

GROUPS = ("left_arm", "right_arm")


@dataclass(frozen=True, slots=True)
class RequestMark:
    session_id: str
    sequence: int
    started_ns: int
    observation_ns: int
    reference_plan: str | None
    submitted_rows: int
    forecast: int = 0


@dataclass(slots=True)
class ExecutionWindow:
    plan_id: str
    start_ns: int
    dt_ns: int
    steps: int
    source_offset: int
    last_row: int = -1

    @property
    def consumed(self) -> int:
        return self.last_row + 1


@dataclass(slots=True)
class PendingSelection:
    prefix: ActionChunk
    full_actions: np.ndarray
    decision: PreparedDecision
    history_revision: int
    offset: int


def json_value(value: Any) -> Any:
    """Represent undefined diagnostics as null; actions and trace must be finite."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class ChunkTrustAhsStrategy(SynchronousChunkStrategy):
    """Synchronous PI-family AHS, with a separately owned immutable configuration."""

    name = "chunktrust_ahs"
    label = "ChunkTrust AHS"
    request_type = InferenceRequest
    hold_last_step = True
    warmup_required_calls = {"ordinary": 2}
    is_rtc = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.options = ChunkTrustOptions.from_mapping(config["inference"]["chunktrust"])
        self._horizon = int(config["policy"]["horizon_policy_steps"])
        self._dt_ns = int(action_interval(config["policy"]) * 1e9)
        if self._horizon != 50 or self._dt_ns < 1:
            raise ValueError(
                "ChunkTrust requires the PI H50 profile and a positive policy interval"
            )
        self._selector = ChunkTrustAdapter(self.options, horizon=self._horizon, action_dim=14)
        self._profile: TraceProfile | None = None
        self._generation = 0
        self._reset_state()

    @property
    def required_sampling_modes(self) -> frozenset[str]:
        return frozenset({"default"})

    @property
    def discard_plans_while_paused(self) -> bool:
        return True

    def validate_capabilities(self, capabilities: PolicyCapabilities) -> None:
        self._profile = validate_capabilities(capabilities, self._config, rtc=self.is_rtc)

    def _reset_state(self) -> None:
        self._selector.reset()
        self._generation += 1
        self._requests: dict[int, RequestMark] = {}
        self._decoded: dict[int, tuple[str, object]] = {}
        self._pending: dict[int, PendingSelection] = {}
        self._active: ExecutionWindow | None = None
        self._full_actions: np.ndarray | None = None
        self._history = np.empty((0, 14), dtype=np.float32)
        self._history_rows: list[dict[str, Any]] = []
        self._history_revision = 0
        self._history_sources = {"submitted": 0, "preview": 0}
        self._events: list[tuple[str, dict[str, object]]] = []

    def reset(self) -> None:
        self._reset_state()

    def _require_profile(self) -> TraceProfile:
        if self._profile is None:
            raise ValueError(
                "ChunkTrust requires validated live trace capabilities before inference"
            )
        return self._profile

    def _remember(self, submission: InferenceSubmission, now_ns: int, forecast: int = 0) -> None:
        request = submission.request
        self._requests[request.request_seq] = RequestMark(
            request.session_id, request.request_seq, now_ns, request.observation_time_ns,
            None if self._active is None else self._active.plan_id,
            self._history_revision, forecast,
        )

    def build_submission(self, **arguments: Any) -> InferenceSubmission | None:
        self._require_profile()
        submission = super().build_submission(**arguments)
        if submission is not None:
            self._remember(submission, arguments["now_ns"])
        return submission

    def decode_action(
        self, raw: object, context: ActionContext, *, adapter: PolicyAdapter,
    ) -> ActionChunk:
        self._require_profile()
        if context.request_seq not in self._requests:
            raise ValueError("ChunkTrust response has no current-generation request")
        chunk = attach_sampling_stats(adapter.decode_action(raw, context), raw)
        self._decoded[context.request_seq] = (chunk.plan_id, raw)
        return chunk

    def select_chunk(self, *, chunk: ActionChunk, now_ns: int) -> ActionChunk:
        """Select after decoding and retain this clock in the proposal snapshot.

        Ordinary AHS starts execution after local work completes. RTC retains its
        shared scoring/commit clock for time-aligned conditioning and cropping.
        """
        decoded = self._decoded.get(chunk.request_seq)
        if decoded is None or decoded[0] != chunk.plan_id:
            raise ValueError("ChunkTrust selection requires its own complete decoded response")
        return self._select(chunk, decoded[1], now_ns)

    def _offset(self, mark: RequestMark) -> int:
        del mark
        return 0

    def _select(self, chunk: ActionChunk, raw: object, now_ns: int) -> ActionChunk:
        profile = self._require_profile()
        mark = self._requests.get(chunk.request_seq)
        if mark is None or mark.observation_ns != chunk.observation_time_ns:
            raise ValueError("ChunkTrust selection request/observation identity mismatch")
        if (
            chunk.action_space != "joint_position" or chunk.source_offset_steps
            or chunk.horizon_steps != self._horizon or chunk.dt_ns != self._dt_ns
            or chunk.runtime_trajectory is not None or chunk.handoff is not None
            or chunk.hold_from_step or set(chunk.groups) != set(GROUPS)
            or any(values.shape != (self._horizon, 7) for values in chunk.groups.values())
        ):
            raise ValueError("ChunkTrust requires a complete untrimmed H50 joint14 prediction")
        offset = self._offset(mark)
        cached = self._pending.get(chunk.request_seq)
        if cached is not None:
            if cached.prefix.plan_id != chunk.plan_id:
                raise ValueError("ChunkTrust request identity was reused for a different plan")
            if cached.history_revision == self._history_revision and cached.offset == offset:
                result = copy_action_chunk(cached.prefix)
                result.metadata = deepcopy(cached.prefix.metadata)
                return result
            self._selector.discard(cached.decision.request_key)
            del self._pending[chunk.request_seq]
        velocity = extract_velocity(raw, profile)
        full_actions = np.concatenate([chunk.groups[name] for name in GROUPS], axis=1).astype(
            np.float32
        )
        if not np.isfinite(full_actions).all():
            raise ValueError("ChunkTrust decoded actions must remain finite in float32")
        if not 0 <= offset < self._horizon:
            raise ValueError("ChunkTrust source suffix is exhausted")
        request_key = f"{self._generation}:{mark.session_id}:{mark.sequence}"
        context = {
            "request_key": request_key, "request_seq": mark.sequence,
            "observation_time_ns": mark.observation_ns, "handoff_time_ns": now_ns,
            "reference_plan_id": mark.reference_plan,
            "source_offset": offset, "full_horizon": self._horizon,
            "history_revision": self._history_revision,
            "timing_profile": "manimux_policy_grid_online" if self.is_rtc else "manimux_serial",
        }
        decision = self._selector.prepare(
            velocity[:, offset:], full_actions[offset:], self._history.copy(),
            request_key=request_key, context=context,
        )
        selected = decision.execution_steps
        prefix = copy_action_chunk(chunk)
        prefix.source_offset_steps = offset
        prefix.groups = {
            name: values[offset:offset + selected].copy() for name, values in chunk.groups.items()
        }
        prefix.metadata["chunktrust"] = json_value({
            **context, "algorithm": self.name, "selected_horizon_steps": selected,
            "available_horizon": self._horizon - offset,
            "history_targets": self._history.copy(), "history_rows": deepcopy(self._history_rows),
            "history_semantics": "submitted_policy_rows_before_interpolation",
            "evidence": decision.info, "selector_state_before": decision.before,
            "selector_state_after": decision.after,
        })
        expected = copy_action_chunk(prefix)
        expected.metadata = deepcopy(prefix.metadata)
        self._pending[chunk.request_seq] = PendingSelection(
            expected, full_actions, decision, self._history_revision, offset,
        )
        return prefix

    def _checked_proposal(
        self, chunk: ActionChunk, response: InferenceResponse,
    ) -> PendingSelection:
        pending = self._pending.get(response.request_seq)
        mark = self._requests.get(response.request_seq)
        if pending is None or mark is None:
            raise ValueError("ChunkTrust requires a current selected proposal")
        expected = pending.prefix
        fields = (
            "request_seq", "plan_id", "observation_time_ns", "created_time_ns", "dt_ns",
            "action_space", "source_offset_steps", "horizon_steps", "hold_from_step",
        )
        if (
            response.session_id != mark.session_id
            or response.observation_time_ns != mark.observation_ns
            or any(getattr(chunk, key) != getattr(expected, key) for key in fields)
            or chunk.runtime_trajectory is not None or chunk.handoff is not None
            or set(chunk.groups) != set(expected.groups)
            or any(not np.array_equal(chunk.groups[name], values)
                   for name, values in expected.groups.items())
            or chunk.metadata.get("chunktrust") != expected.metadata.get("chunktrust")
            or pending.history_revision != self._history_revision
            or pending.decision.revision != self._selector.revision
            or pending.decision.generation != self._selector.generation
        ):
            raise ValueError("ChunkTrust execution prefix differs from its current proposal")
        return pending

    def prepare_chunk(
        self, *, chunk: ActionChunk, response: InferenceResponse, now_ns: int,
    ) -> ActionChunk:
        del now_ns
        self._checked_proposal(chunk, response)
        return chunk

    def on_plan_accepted(
        self, *, chunk: ActionChunk, result: CommitResult,
        response: InferenceResponse, now_ns: int,
    ) -> dict[str, object]:
        pending = self._checked_proposal(chunk, response)
        if not result.accepted or result.trimmed_steps:
            raise ValueError("ChunkTrust acceptance differs from its selected execution prefix")
        old = self._active
        self._selector.commit(pending.decision)
        self._active = ExecutionWindow(
            chunk.plan_id, now_ns, chunk.dt_ns, chunk.horizon_steps, chunk.source_offset_steps,
        )
        self._full_actions = pending.full_actions.copy()
        mark = self._requests.pop(response.request_seq)
        self._pending.pop(response.request_seq)
        self._decoded.pop(response.request_seq, None)
        fields: dict[str, object] = {
            "selected_horizon_steps": chunk.horizon_steps,
            "chunktrust_source_offset": chunk.source_offset_steps,
            "chunktrust_history_rows": len(self._history),
            "chunktrust_unexecuted_old_budget": 0 if old is None else old.steps - old.consumed,
        }
        self._after_accept(mark, now_ns, fields)
        return fields

    def _after_accept(self, mark: RequestMark, now_ns: int, fields: dict[str, object]) -> None:
        del mark, now_ns, fields

    def on_response_rejected(self, response: InferenceResponse) -> None:
        pending = self._pending.pop(response.request_seq, None)
        if pending is not None:
            self._selector.discard(pending.decision.request_key)
        self._requests.pop(response.request_seq, None)
        self._decoded.pop(response.request_seq, None)

    def on_command_feedback(self, feedback: CommandFeedback) -> None:
        if feedback.source == "validated_only":
            return
        if feedback.source not in self._history_sources:
            raise ValueError("unknown ChunkTrust history source")
        active, command = self._active, feedback.command
        if (
            active is None or feedback.reference_plan_id != active.plan_id
            or command.plan_id != active.plan_id or command.monotonic_ns < active.start_ns
        ):
            return
        row = (command.monotonic_ns - active.start_ns) // active.dt_ns
        if not active.last_row < row < active.steps:
            return
        if set(command.groups) != set(GROUPS) or any(
            values.shape != (7,) for values in command.groups.values()
        ):
            raise ValueError("ChunkTrust history requires complete joint14 targets")
        other = "preview" if feedback.source == "submitted" else "submitted"
        if self._history_sources[other]:
            raise ValueError("ChunkTrust preview and formal history require separate instances")
        if row > active.last_row + 1:
            if self.is_rtc:
                raise ValueError("ChunkTrust RTC cannot condition across unsubmitted policy rows")
            self._events.append(("chunktrust_history_rows_skipped", {
                "plan_id": active.plan_id, "first_row": active.last_row + 1,
                "last_row": int(row) - 1,
            }))
        # Feedback identifies an executed policy interval, not the scoring value.
        # RTC can start at a cropped source row; never include expired/unused rows
        # or substitute an interpolated control target for that model prediction.
        if self._full_actions is None:
            raise RuntimeError("ChunkTrust history requires the accepted source prediction")
        target = self._full_actions[active.source_offset + int(row)].copy()
        if not np.isfinite(target).all():
            raise ValueError("ChunkTrust history must be finite in float32")
        self._history = np.concatenate([self._history, target[None]], axis=0)[
            -self.options.history_window:
        ]
        self._history_rows.append({
            "plan_id": active.plan_id, "row": int(row),
            "source_row": active.source_offset + int(row), "source": feedback.source,
            "submitted_time_ns": command.monotonic_ns,
        })
        self._history_rows = self._history_rows[-self.options.history_window:]
        self._history_revision += 1
        self._history_sources[feedback.source] += 1
        active.last_row = int(row)

    def take_runtime_events(self, *, step: int) -> list[tuple[str, dict[str, object]]]:
        events, self._events = self._events, []
        return [(name, {"step": step, **fields}) for name, fields in events]

    def state_dict(self) -> dict[str, Any]:
        """Audit snapshot including external history and the current consumption ledger."""
        return cast(dict[str, Any], json_value({
            "algorithm": self.name, "generation": self._generation,
            "selector": self._selector.snapshot(), "history": self._history.copy(),
            "history_rows": deepcopy(self._history_rows),
            "history_revision": self._history_revision,
            "history_sources": dict(self._history_sources),
            "active": None if self._active is None else asdict(self._active),
            "full_prediction": self._full_actions,
            "pending_requests": sorted(self._pending),
        }))
