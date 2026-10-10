"""Pre-rollout model warmup, isolated from execution and formal plan metrics."""

from __future__ import annotations

from collections import deque

from manimux.clock import Clock
from manimux.policies.worker import PolicyWorkerClient
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.inference import (
    InferenceStrategy,
    build_warmup_submission,
    decode_strategy_action,
    seed_strategy_warmup,
)
from manimux.timing import stage
from manimux.types import ActionChunk, ActionContext, ObservationSnapshot


class PolicyWarmup:
    """Run real requests while paused, then fence them with an acknowledged reset.

    This component has no robot, executor, timeline, or recorder reference. Its
    decoded outputs are offered only as RoboGUI previews, never as executable plans.
    """

    def __init__(
        self, config: dict, *, worker: PolicyWorkerClient, strategy: InferenceStrategy,
        adapter: PolicyAdapter, session_id: str, clock: Clock,
    ) -> None:
        self.config = config
        self.worker = worker
        self.strategy = strategy
        self.adapter = adapter
        self.session_id = session_id
        self.clock = clock
        self.phase = "warming"
        self.completed_requests = 0
        self.latest_branch: str | None = None
        self.error = ""
        self.latency_ns: deque[int] = deque(
            maxlen=config["inference"]["rtc"]["delay_buffer_size"]
        )
        self._recent_ns: deque[int] = deque(maxlen=10)
        self.first_inference_ms: dict[str, float] = {}
        self._request_seq = 0
        self._pending = None
        self._reset_seq: int | None = None
        self._was_paused = True
        self._events: list[tuple[str, dict]] = []
        self._preview: tuple[ActionChunk, float, dict] | None = None

    @property
    def complete(self) -> bool:
        return self.phase == "complete"

    def metadata(self) -> dict:
        return {
            "phase": self.phase,
            "completed_requests": self.completed_requests,
            "latest_branch": self.latest_branch,
            "recent_inference_ms": [round(value / 1e6, 3) for value in self._recent_ns],
            "first_inference_ms": dict(self.first_inference_ms),
            "calibration_sample_count": len(self.latency_ns),
            "latency_calibrated": bool(self.latency_ns),
            "calibration_latency_ns": list(self.latency_ns),
            "error": self.error,
        }

    def take_events(self) -> list[tuple[str, dict]]:
        events, self._events = self._events, []
        return events

    def take_preview(self) -> tuple[ActionChunk, float, dict] | None:
        """Consume the latest display-only output; Start suppresses late previews."""
        preview, self._preview = self._preview, None
        return preview if self.phase == "warming" else None

    def _event(self, name: str, **fields) -> None:
        self._events.append((name, fields))

    def advance(self, *, paused: bool, snapshot: ObservationSnapshot) -> None:
        try:
            self._advance(paused=paused, snapshot=snapshot)
        except Exception as exc:  # IPC failures must remain visible while motion stays paused.
            self.phase = "error"
            self._preview = None
            self.error = f"{type(exc).__name__}: {exc}"
            self._event("warmup_failed", error=self.error)

    def _advance(self, *, paused: bool, snapshot: ObservationSnapshot) -> None:
        start_requested = self._was_paused and not paused
        self._was_paused = paused
        if self.complete:
            return
        if not self.worker.is_alive:
            self.phase = "error"
            self.error = "Policy worker stopped. Finish and prepare a new rollout."
            return
        if start_requested and self.phase in {"warming", "error"}:
            self.phase = "draining"
            self._preview = None
            self.error = ""
            self._event("warmup_start_requested")

        if self._pending is not None:
            with stage("worker_poll"):
                response = self.worker.poll()
            if response is not None:
                request, started_ns, branch = self._pending
                self._pending = None
                error = response.error
                if (
                    response.session_id != self.session_id
                    or response.request_seq != request.request_seq
                ):
                    error = "Unexpected warmup response identity"
                if error is None:
                    try:
                        with stage("policy_decode"):
                            chunk = decode_strategy_action(
                                self.strategy, self.adapter,
                                response.raw_action,
                                ActionContext(
                                    request_seq=response.request_seq,
                                    observation_time_ns=response.observation_time_ns,
                                    created_time_ns=response.finished_time_ns,
                                    measured_state=request.observation.state,
                                    max_source_steps=self.config["policy"]["horizon_policy_steps"],
                                ),
                            )
                        select_chunk = getattr(self.strategy, "select_chunk", None)
                        if callable(select_chunk):
                            chunk = select_chunk(chunk=chunk, now_ns=self.clock.now_ns())
                        if chunk.action_space != "joint_position":
                            raise ValueError("Warmup requires canonical joint-position output")
                    except Exception as exc:  # Keep model/adapter failures visible without motion.
                        error = f"{type(exc).__name__}: {exc}"
                self.error = error or ""
                if error is None:
                    end_ns = self.clock.now_ns()
                    elapsed_ns = max(0, end_ns - min(started_ns, request.observation_time_ns))
                    self._recent_ns.append(elapsed_ns)
                    first_call = branch not in self.first_inference_ms
                    if not first_call:
                        self.latency_ns.append(elapsed_ns)
                        seed_strategy_warmup(self.strategy, latency_ns=list(self.latency_ns))
                    self.completed_requests += 1
                    self.first_inference_ms.setdefault(branch, elapsed_ns / 1e6)
                    if self.phase == "warming":
                        self._preview = (
                            chunk,
                            response.inference_ms,
                            {
                                "warmup_preview": True,
                                "warmup_branch": branch,
                                "warmup_e2e_ms": elapsed_ns / 1e6,
                                "conditioned": branch == "conditioned",
                            },
                        )
                    self._event(
                        "warmup_inference_completed", branch=branch,
                        request_seq=request.request_seq, e2e_ms=elapsed_ns / 1e6,
                        inference_ms=response.inference_ms, first_call=first_call,
                    )
                else:
                    self._event(
                        "warmup_inference_failed", branch=branch,
                        request_seq=request.request_seq, error=error,
                    )

        if self.phase == "draining" and self._pending is None:
            try:
                with stage("worker_submit_reset"):
                    self._reset_seq = self.worker.submit_reset(self.session_id)
                self.phase = "resetting"
            except Exception as exc:
                self.phase = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self._event("warmup_reset_failed", error=self.error)

        if self.phase == "resetting":
            with stage("worker_poll_reset"):
                result = self.worker.poll_reset()
            if result is not None:
                error = result.error
                if result.reset_seq != self._reset_seq or result.session_id != self.session_id:
                    error = "Unexpected warmup reset acknowledgement"
                self.error = error or ""
                self.phase = "error" if error else "complete"
                self._event(
                    "warmup_reset_failed" if error else "warmup_completed", **self.metadata()
                )

        if self.phase == "warming" and self._pending is None:
            started_ns = self.clock.now_ns()
            try:
                # Negative IDs cannot be mistaken for a formal rollout request.
                self._request_seq -= 1
                with stage("policy_schedule"):
                    submission = build_warmup_submission(
                        self.strategy, self.config, session_id=self.session_id,
                        request_seq=self._request_seq, now_ns=started_ns,
                        snapshot=snapshot, adapter=self.adapter,
                        conditioned=bool(self.completed_requests % 2),
                    )
                with stage("policy_prepare"):
                    request = self.adapter.prepare_request(submission.request)
                branch = "conditioned" if submission.event_fields.get("conditioned") else "ordinary"
                with stage("worker_submit"):
                    self.worker.submit_latest(request)
                self._pending = (request, started_ns, branch)
                self.latest_branch = branch
                self._event(
                    "warmup_inference_submitted", request_seq=request.request_seq, branch=branch,
                    horizon_steps=self.config["policy"]["horizon_policy_steps"],
                )
            except Exception as exc:
                self.phase = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self._event("warmup_inference_failed", error=self.error)
