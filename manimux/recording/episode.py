from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import zarr

from manimux.timing import LoopTiming, timed
from manimux.types import (
    ActionChunk,
    ActionHorizon,
    GroupVector,
    RobotState,
    SensorFrame,
    copy_action_chunk,
)

from .video import AsyncVideoRecorder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TickRecord:
    monotonic_ns: int
    state: GroupVector
    scheduled: GroupVector
    command: GroupVector
    plan_id: str | None
    inference_ms: float | None
    camera_times_ns: dict[str, int]
    sent_commands: dict[str, dict]


@dataclass(slots=True)
class _PlanRecord:
    canonical_raw: ActionChunk
    infra_output: ActionChunk
    committed: ActionHorizon


class EpisodeRecorder:
    """Milestone-0 local recorder: stream events, buffer numeric tick data, finalize Zarr."""

    def __init__(
        self,
        run_dir: Path,
        episode_id: str,
        group_dims: dict[str, int],
        metadata: dict[str, object],
        *,
        video_fps: float = 0.0,
        video_codec: str = "mp4v",
        video_queue_size: int = 8,
        control_timing: LoopTiming | None = None,
    ) -> None:
        self._partial_dir = run_dir / f"{episode_id}.partial"
        self._final_dir = run_dir / episode_id
        self._partial_dir.mkdir(parents=True, exist_ok=False)
        self._group_dims = dict(group_dims)
        self._ticks: list[_TickRecord] = []
        self._plans: list[_PlanRecord] = []
        self._events_path = self._partial_dir / "events.jsonl"
        self._events: TextIO = self._events_path.open("a", encoding="utf-8")
        self._metadata = dict(metadata)
        self._metadata_path = self._partial_dir / "meta.json"
        self._control_timing = control_timing
        self._video = AsyncVideoRecorder(
            self._partial_dir,
            fps=video_fps,
            codec=video_codec,
            queue_size=video_queue_size,
        )
        self._write_json(self._metadata_path, self._metadata)
        self.event("episode_started", episode_id=episode_id)

    @property
    def final_dir(self) -> Path:
        return self._final_dir

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @timed("recording.event")
    def event(self, kind: str, **fields: object) -> None:
        payload = {"kind": kind, **fields}
        self._events.write(json.dumps(payload, sort_keys=True) + "\n")
        self._events.flush()

    @timed("recording.metadata")
    def update_metadata(self, **fields: object) -> None:
        self._metadata.update(fields)
        self._write_json(self._metadata_path, self._metadata)

    @staticmethod
    def _copy_horizon(horizon: ActionHorizon) -> ActionHorizon:
        return ActionHorizon(
            start_time_ns=horizon.start_time_ns,
            dt_ns=horizon.dt_ns,
            plan_id=horizon.plan_id,
            groups={name: values.copy() for name, values in horizon.groups.items()},
            hold_groups=horizon.hold_groups,
            tracking_groups=(
                None
                if horizon.tracking_groups is None
                else {name: value.copy() for name, value in horizon.tracking_groups.items()}
            ),
            observation_time_ns=horizon.observation_time_ns,
        )

    @timed("recording.plan")
    def record_plan(
        self,
        *,
        canonical_raw: ActionChunk,
        infra_output: ActionChunk,
        committed: ActionHorizon,
    ) -> None:
        self._plans.append(
            _PlanRecord(
                canonical_raw=copy_action_chunk(canonical_raw),
                infra_output=copy_action_chunk(infra_output),
                committed=self._copy_horizon(committed),
            )
        )

    @timed("recording.tick")
    def record_tick(
        self,
        *,
        monotonic_ns: int,
        state: RobotState,
        scheduled: GroupVector,
        command: GroupVector,
        plan_id: str | None,
        inference_ms: float | None,
        camera_times_ns: dict[str, int],
        frames: dict[str, SensorFrame] | None = None,
        sent_commands: dict[str, dict] | None = None,
    ) -> None:
        self._ticks.append(
            _TickRecord(
                monotonic_ns=monotonic_ns,
                state={name: value.copy() for name, value in state.groups.items()},
                scheduled={name: value.copy() for name, value in scheduled.items()},
                command={name: value.copy() for name, value in command.items()},
                plan_id=plan_id,
                inference_ms=inference_ms,
                camera_times_ns=dict(camera_times_ns),
                sent_commands={
                    name: {key: value.copy() if isinstance(value, np.ndarray) else value
                           for key, value in sample.items()}
                    for name, sample in (sent_commands or {}).items()
                },
            )
        )
        self._video.submit(frames or {})

    def _write_zarr(self) -> None:
        root = zarr.open_group(str(self._partial_dir / "data.zarr"), mode="w")
        ticks = root.create_group("ticks")
        ticks.create_dataset(
            "monotonic_ns",
            data=np.asarray([record.monotonic_ns for record in self._ticks], dtype=np.int64),
        )
        ticks.create_dataset(
            "inference_ms",
            data=np.asarray(
                [
                    np.nan if record.inference_ms is None else record.inference_ms
                    for record in self._ticks
                ],
                dtype=np.float64,
            ),
        )
        plan_ids = ["" if record.plan_id is None else record.plan_id for record in self._ticks]
        ticks.create_dataset("plan_id", data=np.asarray(plan_ids, dtype="U64"))
        for stage in ("state", "scheduled", "command"):
            stage_group = ticks.create_group(stage)
            for name, dim in self._group_dims.items():
                stage_values = [getattr(record, stage)[name] for record in self._ticks]
                array = (
                    np.stack(stage_values) if stage_values else np.empty((0, dim), dtype=np.float64)
                )
                stage_group.create_dataset(name, data=array)

        sent = ticks.create_group("sent_command")
        sent.attrs.update({
            "source": "latest complete i2rt MIT CAN send cycle, sampled per recording tick",
            "position": "decoded wire target mapped to joint radians and normalized gripper",
            "motor_position": "decoded MIT position in motor radians",
            "send_time_ns": "Unix ns before last successful host bus.send per motor",
            "send_monotonic_ns": "monotonic ns before that host bus.send",
            "send_end_monotonic_ns": "monotonic ns after that host bus.send returns",
            "send_count": "successful host sends including retries in that motor transaction",
            "sequence": "SDK-local complete-cycle counter; repeated rows are cached samples",
            "missing": "valid=false, position=NaN, integer fields=-1; never substitute command",
            "scope": "host send evidence, not hardware receive timestamps or a full CAN log",
        })
        for name, dim in self._group_dims.items():
            group = sent.create_group(name)
            samples = [record.sent_commands.get(name) for record in self._ticks]
            group.create_dataset(
                "valid", data=np.asarray([s is not None for s in samples], dtype=bool)
            )
            group.create_dataset("sequence", data=np.asarray([
                -1 if s is None else s["sequence"] for s in samples
            ], dtype=np.int64))
            for key in ("position", "motor_position", "send_time_ns", "send_monotonic_ns",
                        "send_end_monotonic_ns", "send_count"):
                floating = key in {"position", "motor_position"}
                dtype = np.float64 if floating else np.int64
                values = [np.full(dim, np.nan if floating else -1, dtype=dtype)
                          if s is None else np.asarray(s[key], dtype=dtype) for s in samples]
                group.create_dataset(
                    key, data=np.stack(values) if values else np.empty((0, dim), dtype=dtype)
                )

        camera_names = sorted({name for record in self._ticks for name in record.camera_times_ns})
        camera_group = ticks.create_group("camera_time_ns")
        for name in camera_names:
            camera_group.create_dataset(
                name,
                data=np.asarray(
                    [record.camera_times_ns.get(name, -1) for record in self._ticks],
                    dtype=np.int64,
                ),
            )

        plans = root.create_group("plans")
        for index, record in enumerate(self._plans):
            plan = plans.create_group(f"{index:06d}")
            plan.attrs.update(
                {
                    "plan_id": record.infra_output.plan_id,
                    "request_seq": record.infra_output.request_seq,
                }
            )
            for stage_name, chunk in (
                ("canonical_raw", record.canonical_raw),
                ("infra_output", record.infra_output),
            ):
                stage = plan.create_group(stage_name)
                stage.attrs.update(
                    {
                        "observation_time_ns": chunk.observation_time_ns,
                        "created_time_ns": chunk.created_time_ns,
                        "action_space": chunk.action_space,
                        "dt_ns": chunk.dt_ns,
                        "source_offset_steps": chunk.source_offset_steps,
                        "hold_from_step": dict(chunk.hold_from_step),
                        "metadata": dict(chunk.metadata),
                    }
                )
                for name, plan_values in chunk.groups.items():
                    stage.create_dataset(name, data=plan_values)
            committed = plan.create_group("committed")
            committed.attrs.update(
                {
                    "start_time_ns": record.committed.start_time_ns,
                    "observation_time_ns": record.committed.observation_time_ns,
                    "dt_ns": record.committed.dt_ns,
                    "plan_id": record.committed.plan_id,
                    "action_space": record.infra_output.action_space,
                    "time_trimmed_steps": record.infra_output.metadata.get("time_trimmed_steps", 0),
                    "handoff_skipped_steps": record.infra_output.metadata.get(
                        "handoff_skipped_steps", 0
                    ),
                }
            )
            for name, plan_values in record.committed.groups.items():
                committed.create_dataset(name, data=plan_values)

    def _save_control_timing(self) -> dict:
        if self._control_timing is None:
            return {}
        try:
            summary = self._control_timing.write(self._partial_dir)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Control timing could not be saved: %s", error, exc_info=True)
            return {"control_timing": {"status": "error", "error": error}}
        return {"control_timing": {"status": "disabled" if summary is None else "saved"}}

    def finish(
        self,
        *,
        success: bool,
        terminal_reason: str,
        steps: int,
        wall_time_s: float,
    ) -> Path:
        timing_status = self._save_control_timing()
        self.event(
            "episode_finished",
            success=success,
            terminal_reason=terminal_reason,
            steps=steps,
        )
        self._events.close()
        video = self._video.close()
        self._write_zarr()
        self._write_json(
            self._partial_dir / "result.json",
            {
                "success": success,
                "terminal_reason": terminal_reason,
                "steps": steps,
                "wall_time_s": wall_time_s,
                **timing_status,
                "video_recording": {
                    "enabled": video.enabled,
                    "frames_written": video.frames_written,
                    "dropped_bundles": video.dropped_bundles,
                    "error": video.error,
                },
            },
        )
        self._partial_dir.rename(self._final_dir)
        return self._final_dir

    def abort(self, reason: str, *, detail: str = "") -> None:
        if self._events.closed:
            return
        timing_status = self._save_control_timing()
        self.event("episode_aborted", terminal_reason=reason, detail=detail)
        self._events.close()
        video = self._video.close()
        self._write_zarr()
        self._write_json(
            self._partial_dir / "result.json",
            {
                "success": False,
                "terminal_reason": reason,
                "detail": detail,
                "steps": len(self._ticks),
                "incomplete": True,
                **timing_status,
                "video_recording": {
                    "enabled": video.enabled,
                    "frames_written": video.frames_written,
                    "dropped_bundles": video.dropped_bundles,
                    "error": video.error,
                },
            },
        )
