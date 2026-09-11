from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import zarr

from manimux.types import (
    ActionChunk,
    ActionHorizon,
    GroupVector,
    RobotState,
    SensorFrame,
    copy_action_chunk,
)

from .video import AsyncVideoRecorder


@dataclass(slots=True)
class _TickRecord:
    monotonic_ns: int
    state: GroupVector
    scheduled: GroupVector
    optimized: GroupVector
    command: GroupVector
    plan_id: str | None
    inference_ms: float | None
    camera_times_ns: dict[str, int]


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

    def event(self, kind: str, **fields: object) -> None:
        payload = {"kind": kind, **fields}
        self._events.write(json.dumps(payload, sort_keys=True) + "\n")
        self._events.flush()

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

    def record_tick(
        self,
        *,
        monotonic_ns: int,
        state: RobotState,
        scheduled: GroupVector,
        optimized: GroupVector,
        command: GroupVector,
        plan_id: str | None,
        inference_ms: float | None,
        camera_times_ns: dict[str, int],
        frames: dict[str, SensorFrame] | None = None,
    ) -> None:
        self._ticks.append(
            _TickRecord(
                monotonic_ns=monotonic_ns,
                state={name: value.copy() for name, value in state.groups.items()},
                scheduled={name: value.copy() for name, value in scheduled.items()},
                optimized={name: value.copy() for name, value in optimized.items()},
                command={name: value.copy() for name, value in command.items()},
                plan_id=plan_id,
                inference_ms=inference_ms,
                camera_times_ns=dict(camera_times_ns),
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
        for stage in ("state", "scheduled", "optimized", "command"):
            stage_group = ticks.create_group(stage)
            for name, dim in self._group_dims.items():
                stage_values = [getattr(record, stage)[name] for record in self._ticks]
                array = (
                    np.stack(stage_values) if stage_values else np.empty((0, dim), dtype=np.float64)
                )
                stage_group.create_dataset(name, data=array)

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
                }
            )
            for name, plan_values in record.committed.groups.items():
                committed.create_dataset(name, data=plan_values)

    def finish(
        self,
        *,
        success: bool,
        terminal_reason: str,
        steps: int,
        wall_time_s: float,
    ) -> Path:
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

    def abort(self, reason: str) -> None:
        if self._events.closed:
            return
        self.event("episode_aborted", terminal_reason=reason)
        self._events.close()
        video = self._video.close()
        self._write_zarr()
        self._write_json(
            self._partial_dir / "result.json",
            {
                "success": False,
                "terminal_reason": reason,
                "steps": len(self._ticks),
                "incomplete": True,
                "video_recording": {
                    "enabled": video.enabled,
                    "frames_written": video.frames_written,
                    "dropped_bundles": video.dropped_bundles,
                    "error": video.error,
                },
            },
        )
