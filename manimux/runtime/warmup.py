"""Reset-fenced dry run of the selected configuration, without action dispatch."""

from collections import Counter, deque
from dataclasses import dataclass, replace

from manimux.runtime.inference import InferenceTimingError, decode_strategy_action
from manimux.runtime.preview import PreviewPlanner
from manimux.types import copy_action_chunk


@dataclass
class PendingPreview:
    request: object
    started_ns: int
    branch: str
    fields: dict
    decode_started_ns: int = 0


class PolicyWarmup:
    """Use real observations and the rollout planner; discard predicted commands.

    Negative transport/GUI IDs fence previews from formal plans. Strategy, adapter,
    timeline, scheduler, executor and safety history are private to this dry run.
    The process decoder is borrowed exclusively until the reset acknowledgement.
    There is no robot/recorder/command sink in this component.
    """

    def __init__(
        self,
        config,
        *,
        worker,
        strategy,
        adapter,
        session_id,
        clock,
        executor,
        kinematics,
        safety,
        decoder=None,
    ):
        self.config, self.worker, self.strategy, self.adapter = config, worker, strategy, adapter
        self.session_id, self.clock = session_id, clock
        self.planner = PreviewPlanner(
            config,
            strategy=strategy,
            adapter=adapter,
            executor=executor,
            kinematics=kinematics,
            safety=safety,
            decoder=decoder,
            session_id=session_id,
            clock=clock,
        )
        self.decoder = decoder
        self.phase = "idle"
        self.completed_requests = 0
        self.latest_branch = None
        self.error = ""
        inference = config["inference"]
        delay_config = inference.get(inference["algorithm"], inference["rtc"])
        self.latency_ns = deque(
            maxlen=delay_config.get("delay_buffer_size", inference["rtc"]["delay_buffer_size"])
        )
        self._required_calls = dict(getattr(strategy, "warmup_required_calls", {"ordinary": 1}))
        self._branch_calls = Counter()
        self._recent_ns = deque(maxlen=10)
        self.first_inference_ms = {}
        self._pending = {}
        self._waiting_response = None
        self._decoding_seq = None
        self._reset_seq = None
        self._was_paused = True
        self._events = []
        self._preview = None
        self._execution_ms = None
        self._ik_status = None

    def activate(self):
        if self.phase == "idle":
            self.phase = "warming"
            self._was_paused = True
            self._event("warmup_activated")

    @property
    def complete(self):
        return self.phase == "complete"

    @property
    def ready(self):
        return all(
            self._branch_calls[branch] >= count for branch, count in self._required_calls.items()
        )

    def metadata(self):
        return {
            "phase": self.phase,
            "completed_requests": self.completed_requests,
            "latest_branch": self.latest_branch,
            "required_branch_calls": dict(self._required_calls),
            "completed_branch_calls": dict(self._branch_calls),
            "branches_ready": self.ready,
            "recent_inference_ms": [round(value / 1e6, 3) for value in self._recent_ns],
            "first_inference_ms": dict(self.first_inference_ms),
            "calibration_sample_count": len(self.latency_ns),
            "latency_calibrated": bool(self.latency_ns),
            "calibration_latency_ns": list(self.latency_ns),
            "error": self.error,
            "execution_ms": self._execution_ms,
            "ik_status": self._ik_status,
            "configured_planning": True,
            **self.planner.metadata(),
        }

    def take_events(self):
        events, self._events = self._events, []
        return events

    def take_preview(self):
        preview, self._preview = self._preview, None
        return preview if self.phase == "warming" else None

    def _event(self, name, **fields):
        self._events.append((name, fields))

    def advance(self, *, paused, snapshot):
        try:
            self._advance(paused=paused, snapshot=snapshot)
        except InferenceTimingError:
            raise
        except Exception as exc:
            self.phase = "error"
            self._preview = None
            self.error = f"{type(exc).__name__}: {exc}"
            self._event("warmup_failed", error=self.error)

    def _fail_response(self, response, error):
        job = self._pending.pop(response.request_seq)
        self.strategy.on_response_rejected(response)
        self.error = error
        self._event(
            "warmup_inference_failed",
            branch=job.branch,
            request_seq=-(response.request_seq + 1),
            error=error,
        )
        if self.phase == "draining" and not self.ready:
            self.phase = "error"

    def _finish_response(self, response, snapshot, decoded=None):
        job = self._pending[response.request_seq]
        try:
            reason = self.planner.rejection_reason(response)
            if reason:
                raise ValueError(reason)
            if decoded is None:
                context = self.planner._decode_context(
                    response,
                    snapshot.state,
                    self.clock.now_ns(),
                    job.request.observation.state,
                )
                chunk = decode_strategy_action(
                    self.strategy, self.adapter, response.raw_action, context
                )
            else:
                chunk = decoded
                self.planner._decode_forecast.observe(chunk.metadata["decode_stage_ms"])
            decode_finished_ns = self.clock.now_ns()
            chunk, result, fields = self.planner.accept(chunk, response, snapshot.state)
            if not result.accepted:
                raise ValueError(result.reason)
        except (TypeError, ValueError) as exc:
            self._fail_response(response, f"{type(exc).__name__}: {exc}")
            return
        self._pending.pop(response.request_seq)
        end_ns = self.clock.now_ns()
        elapsed_ns = max(0, end_ns - min(job.started_ns, job.request.observation_time_ns))
        breakdown = {
            "observation_age_ms": max(0, job.started_ns - job.request.observation_time_ns) / 1e6,
            "request_to_decode_ms": (job.decode_started_ns - job.started_ns) / 1e6,
            "decode_ms": (decode_finished_ns - job.decode_started_ns) / 1e6,
            "post_decode_ms": (end_ns - decode_finished_ns) / 1e6,
        }
        first_call = job.branch not in self.first_inference_ms
        if not first_call:
            self.latency_ns.append(elapsed_ns)
        self.first_inference_ms.setdefault(job.branch, elapsed_ns / 1e6)
        self._recent_ns.append(elapsed_ns)
        self.completed_requests += 1
        self._branch_calls[job.branch] += 1
        self.error = ""
        if self.phase == "warming":
            display = copy_action_chunk(chunk)
            display.request_seq = -(response.request_seq + 1)
            self._preview = (
                display,
                response.inference_ms,
                {
                    **job.fields,
                    **fields,
                    "warmup_preview": True,
                    "warmup_branch": job.branch,
                    "warmup_e2e_ms": elapsed_ns / 1e6,
                    "warmup_timing": breakdown,
                    "configured_planning": True,
                    "conditioned": job.branch == "conditioned",
                    "trimmed_steps": result.trimmed_steps,
                    "time_trimmed_steps": result.time_trimmed_steps,
                    "handoff_skipped_steps": result.handoff_skipped_steps,
                    "committed_horizon_steps": (
                        self.planner._timeline.active_horizon().horizon_steps
                    ),
                },
            )
        self._event(
            "warmup_inference_completed",
            branch=job.branch,
            request_seq=-(response.request_seq + 1),
            e2e_ms=elapsed_ns / 1e6,
            inference_ms=response.inference_ms,
            first_call=first_call,
            **breakdown,
        )

    def _poll(self, snapshot):
        """Drain model/decoder output; return whether this tick consumed a response."""
        processed = False
        decoded = self.decoder.poll() if self.decoder is not None else None
        if decoded is not None:
            self._decoding_seq = None
            processed = True
            response = decoded.response
            if decoded.error:
                self._fail_response(response, decoded.error)
            else:
                self._finish_response(response, snapshot, decoded.chunk)
        wire = self.worker.poll() if self._pending else None
        if wire is not None:
            processed = True
            seq = -wire.request_seq - 1
            if wire.request_seq >= 0 or seq not in self._pending:
                raise ValueError("Unexpected warmup response identity")
            response = replace(wire, request_seq=seq)
            if self._waiting_response is not None:
                self._fail_response(self._waiting_response, "superseded_pending_response")
            self._waiting_response = response
            # Multi-inflight transport coalesces uploads. Earlier unreturned
            # jobs cannot later become accepted plans, and must not block Start.
            for old_seq in list(self._pending):
                if old_seq < seq and old_seq != self._decoding_seq:
                    self._pending.pop(old_seq)
        if self._waiting_response is not None and self.planner.replacement_ready():
            if self.decoder is not None and self.decoder.busy:
                return processed
            response, self._waiting_response = self._waiting_response, None
            job = self._pending[response.request_seq]
            job.decode_started_ns = self.clock.now_ns()
            reason = self.planner.rejection_reason(response)
            if reason:
                self._fail_response(response, reason)
            elif self.decoder is not None:
                context = self.planner._decode_context(
                    response,
                    snapshot.state,
                    self.clock.now_ns(),
                    job.request.observation.state,
                )
                self.decoder.submit(response, context, job.request.deadline_ns)
                self._decoding_seq = response.request_seq
            else:
                self._finish_response(response, snapshot)
            processed = True
        return processed

    def _advance(self, *, paused, snapshot):
        loop_started_ns = self.clock.now_ns()
        start_requested = self._was_paused and not paused
        self._was_paused = paused
        if self.complete or self.phase == "idle":
            return
        if not self.worker.is_alive:
            raise RuntimeError("Policy worker stopped. Finish and prepare a new rollout.")
        if start_requested and self.phase in {"warming", "error"}:
            self.phase = "draining"
            self._preview = None
            self.error = ""
            self._event("warmup_start_requested")
        if self.phase in {"warming", "draining"}:
            self.planner.initialize(snapshot.state)
            response_processed = self._poll(snapshot)
            # Schedule before execution, as in rollout. Never reuse the snapshot
            # captured before a blocking decode for the next model submission.
            should_submit = self.phase == "warming" or not self.ready
            if should_submit and not response_processed:
                started_ns = self.clock.now_ns()
                submission = self.planner.submission(snapshot, in_flight=bool(self._pending))
                if submission is not None:
                    request = self.adapter.prepare_request(submission.request)
                    self.worker.submit_latest(
                        replace(request, request_seq=-(request.request_seq + 1))
                    )
                    self.planner.submitted(request, started_ns)
                    fields = {**self.planner.metadata(), **submission.event_fields}
                    fields["active_chunk_index"] = fields.pop("chunk_index")
                    branch = "conditioned" if fields.get("conditioned") else "ordinary"
                    if branch == "conditioned":
                        fields["conditioned_overlap_steps"] = max(
                            0,
                            self.config["policy"]["horizon_policy_steps"]
                            - int(fields.get("executed_steps", 0)),
                        )
                        fields["frozen_steps"] = int(fields.get("forecast_delay", 0))
                    self._pending[request.request_seq] = PendingPreview(
                        request, started_ns, branch, fields
                    )
                    self.latest_branch = branch
                    self._event(
                        "warmup_inference_submitted",
                        request_seq=-(request.request_seq + 1),
                        branch=branch,
                        horizon_steps=self.config["policy"]["horizon_policy_steps"],
                        **fields,
                    )
            self._execution_ms, self._ik_status = self.planner.tick(
                snapshot.state, loop_started_ns=loop_started_ns
            )
            for kind, fields in self.strategy.take_runtime_events(step=self.planner.steps):
                self._event("warmup_strategy_event", strategy_event=kind, **fields)
        if self.phase == "draining" and not self._pending and self.ready:
            self._reset_seq = self.worker.submit_reset(self.session_id)
            self.phase = "resetting"
        if self.phase == "resetting":
            result = self.worker.poll_reset()
            if result is not None:
                error = result.error
                if result.reset_seq != self._reset_seq or result.session_id != self.session_id:
                    error = "Unexpected warmup reset acknowledgement"
                self.error = error or ""
                self.phase = "error" if error else "complete"
                if not error:
                    self.strategy.reset()
                self._event(
                    "warmup_reset_failed" if error else "warmup_completed", **self.metadata()
                )
