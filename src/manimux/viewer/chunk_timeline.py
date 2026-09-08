"""Pure state and HTML rendering for the live action-chunk timeline."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChunkLane:
    chunk_id: int | None = None
    horizon_steps: int = 0
    committed_steps: int = 0
    cursor: int = 0
    trimmed_steps: int = 0
    overlap_steps: int = 0
    frozen_steps: int = 0
    superseded_steps: int = 0
    inference_ms: float | None = None
    state: str = "empty"
    conditioned: bool = False
    source_chunk_id: int | None = None
    condition_from_index: int | None = None
    latency_from_index: int | None = None
    gripper_closed_steps: tuple[bool, ...] = ()


class ChunkTimelineView:
    """Reduce viewer wire messages into two alternating chunk lanes."""

    def __init__(self) -> None:
        self.runtime = "waiting"
        self.lanes = [ChunkLane(), ChunkLane()]
        self.active_chunk_id: int | None = None
        self.pending_chunk_id: int | None = None
        self.episode_open = False
        self.service_id = ""
        self._lane_by_chunk: dict[int, int] = {}

    def _lane_index(self, chunk_id: int) -> int:
        existing = self._lane_by_chunk.get(chunk_id)
        if existing is not None:
            return existing
        occupied = [index for index, lane in enumerate(self.lanes) if lane.chunk_id is not None]
        if not occupied:
            index = 0
        elif self.active_chunk_id is not None and self.active_chunk_id in self._lane_by_chunk:
            index = 1 - self._lane_by_chunk[self.active_chunk_id]
        else:
            index = 1 - occupied[-1]
        replaced = self.lanes[index].chunk_id
        if replaced is not None:
            self._lane_by_chunk.pop(replaced, None)
        self._lane_by_chunk[chunk_id] = index
        return index

    def reset(self, runtime: str = "waiting") -> None:
        self.runtime = runtime
        self.lanes = [ChunkLane(), ChunkLane()]
        self.active_chunk_id = None
        self.pending_chunk_id = None
        self.episode_open = False
        self._lane_by_chunk = {}

    def update(self, message: dict[str, Any]) -> None:
        kind = str(message.get("kind", ""))
        if kind == "event":
            self._update_event(message)
        elif kind == "plan":
            self._update_plan(message)
        elif kind == "state":
            self._update_state(message)

    def _update_event(self, message: dict[str, Any]) -> None:
        event = str(message.get("event", ""))
        metadata = message.get("metadata") or {}
        if event == "runtime_service_ready":
            incoming_service_id = str(metadata.get("run_dir", ""))
            if not self.episode_open or (
                incoming_service_id and incoming_service_id != self.service_id
            ):
                self.reset(str(metadata.get("runtime", self.runtime)))
            if incoming_service_id:
                self.service_id = incoming_service_id
            return
        if event == "episode_started":
            self.reset(str(metadata.get("runtime", self.runtime)))
            self.episode_open = True
            incoming_service_id = str(metadata.get("run_dir", ""))
            if incoming_service_id:
                self.service_id = incoming_service_id
            return
        if event == "episode_finished":
            self.episode_open = False
            self.pending_chunk_id = None
            for lane in self.lanes:
                if lane.state == "pending":
                    lane.state = "empty"
            return
        if event == "inference_submitted":
            raw_chunk_id = message.get("chunk_id", metadata.get("request_seq"))
            if raw_chunk_id is None:
                return
            chunk_id = int(raw_chunk_id)
            horizon = max(1, int(metadata.get("horizon_steps", 1)))
            overlap = max(0, int(metadata.get("conditioned_overlap_steps", 0)))
            frozen = max(0, int(metadata.get("frozen_steps", 0)))
            conditioned = bool(metadata.get("conditioned", False))
            active_chunk_id = metadata.get("active_chunk_id")
            source_chunk_id = active_chunk_id if conditioned else None
            source_index_is_raw = "executed_steps" in metadata
            source_index = max(
                0,
                int(
                    metadata["executed_steps"]
                    if source_index_is_raw
                    else metadata.get("active_chunk_index", 0)
                ),
            )
            if active_chunk_id is not None:
                source_lane_index = self._lane_by_chunk.get(int(active_chunk_id))
                if source_lane_index is not None:
                    source_lane = self.lanes[source_lane_index]
                    source_lane.latency_from_index = min(
                        source_lane.horizon_steps,
                        source_index
                        if source_index_is_raw
                        else source_lane.trimmed_steps + source_index,
                    )
                    if conditioned:
                        source_lane.condition_from_index = source_index
            lane = ChunkLane(
                chunk_id=chunk_id,
                horizon_steps=horizon,
                committed_steps=horizon,
                cursor=0,
                overlap_steps=min(overlap, horizon),
                frozen_steps=min(frozen, horizon),
                state="pending",
                conditioned=conditioned,
                source_chunk_id=(
                    None if source_chunk_id is None else int(source_chunk_id)
                ),
            )
            self.lanes[self._lane_index(chunk_id)] = lane
            self.pending_chunk_id = chunk_id
            self.runtime = str(metadata.get("runtime", self.runtime))
            return
        if event in {"inference_rejected", "plan_rejected"}:
            raw_chunk_id = message.get("chunk_id")
            if raw_chunk_id is None:
                return
            chunk_id = int(raw_chunk_id)
            lane_index = self._lane_by_chunk.get(chunk_id)
            if lane_index is None:
                return
            lane = self.lanes[lane_index]
            if lane.chunk_id == chunk_id:
                lane.state = "rejected"
            if self.pending_chunk_id == chunk_id:
                self.pending_chunk_id = None

    def _update_plan(self, message: dict[str, Any]) -> None:
        metadata = message.get("metadata") or {}
        chunk_id = int(message.get("chunk_id", 0))
        previous_chunk_id = metadata.get("previous_chunk_id")
        conditioned = bool(metadata.get("conditioned", False))
        if previous_chunk_id is not None:
            previous_id = int(previous_chunk_id)
            previous_lane_index = self._lane_by_chunk.get(previous_id)
            if previous_lane_index is not None:
                previous_lane = self.lanes[previous_lane_index]
                previous_lane.cursor = max(
                    previous_lane.cursor,
                    int(metadata.get("previous_chunk_index", previous_lane.cursor)),
                )
                previous_lane.superseded_steps = max(
                    0, int(metadata.get("superseded_steps", 0))
                )
                if (
                    previous_lane.latency_from_index is None
                    and "active_chunk_index" in metadata
                ):
                    previous_lane.latency_from_index = min(
                        previous_lane.horizon_steps,
                        previous_lane.trimmed_steps
                        + max(0, int(metadata["active_chunk_index"])),
                    )
                if conditioned and previous_lane.condition_from_index is None:
                    previous_lane.condition_from_index = max(
                        0,
                        int(
                            metadata.get(
                                "executed_steps",
                                previous_lane.condition_from_index
                                if previous_lane.condition_from_index is not None
                                else previous_lane.trimmed_steps + previous_lane.cursor,
                            )
                        ),
                    )
                previous_lane.state = "source" if conditioned else "retired"

        committed_steps = len(message.get("actions", []))
        raw_horizon = max(
            committed_steps,
            int(metadata.get("raw_horizon_steps", committed_steps)),
        )
        trimmed = min(raw_horizon, max(0, int(metadata.get("trimmed_steps", 0))))
        overlap = min(
            raw_horizon,
            max(0, int(metadata.get("conditioned_overlap_steps", 0))),
        )
        frozen = min(raw_horizon, max(0, int(metadata.get("frozen_steps", 0))))
        committed_gripper = tuple(
            bool(value) for value in metadata.get("gripper_closed_steps", ())
        )
        raw_gripper = (
            (False,) * trimmed + committed_gripper[:committed_steps]
        )[:raw_horizon]
        if len(raw_gripper) < raw_horizon:
            raw_gripper += (False,) * (raw_horizon - len(raw_gripper))
        lane = ChunkLane(
            chunk_id=chunk_id,
            horizon_steps=raw_horizon,
            committed_steps=committed_steps,
            cursor=0,
            trimmed_steps=trimmed,
            overlap_steps=overlap,
            frozen_steps=frozen,
            inference_ms=float(message.get("inference_ms", 0.0)),
            state="active",
            conditioned=conditioned,
            source_chunk_id=(
                None if previous_chunk_id is None else int(previous_chunk_id)
            ),
            gripper_closed_steps=raw_gripper,
        )
        self.lanes[self._lane_index(chunk_id)] = lane
        self.active_chunk_id = chunk_id
        if self.pending_chunk_id == chunk_id:
            self.pending_chunk_id = None
        self.runtime = str(metadata.get("runtime", self.runtime))

    def _update_state(self, message: dict[str, Any]) -> None:
        active_chunk_id = message.get("active_chunk_id")
        if active_chunk_id is None:
            return
        chunk_id = int(active_chunk_id)
        lane_index = self._lane_by_chunk.get(chunk_id)
        if lane_index is None:
            return
        lane = self.lanes[lane_index]
        if lane.chunk_id != chunk_id:
            return
        lane.cursor = min(lane.committed_steps, max(0, int(message.get("chunk_index", 0))))
        lane.state = "active"
        self.active_chunk_id = chunk_id

    @staticmethod
    def _cell_state(lane: ChunkLane, index: int) -> str:
        if lane.state == "pending":
            if index < lane.frozen_steps:
                return "latency pending"
            if index < lane.overlap_steps:
                return "overlap pending"
            return "pending"
        if lane.state == "rejected":
            return "rejected"
        if index < lane.trimmed_steps:
            return "latency-trimmed"
        committed_index = index - lane.trimmed_steps
        raw_cursor = lane.trimmed_steps + lane.cursor
        if (
            lane.latency_from_index is not None
            and lane.latency_from_index <= index < raw_cursor
        ):
            return "latency"
        if committed_index < lane.cursor:
            return "executed"
        if (
            lane.state == "active"
            and committed_index == lane.cursor
            and lane.cursor < lane.committed_steps
        ):
            return "current"
        if lane.condition_from_index is not None and index >= lane.condition_from_index:
            return "condition-source"
        if lane.state == "retired" and committed_index >= lane.cursor:
            return "superseded"
        if index < lane.frozen_steps:
            return "frozen"
        if index < lane.overlap_steps:
            return "overlap"
        return "future"

    @staticmethod
    def _lane_summary(lane: ChunkLane) -> str:
        if lane.chunk_id is None:
            return "waiting"
        if lane.state == "pending":
            return "sampling"
        if lane.state == "rejected":
            return "rejected"
        if lane.inference_ms is not None:
            return f"{lane.inference_ms:.0f} ms"
        return lane.state

    def _handoff_html(self) -> str:
        target_index = next(
            (
                index
                for index, lane in enumerate(self.lanes)
                if lane.conditioned
                and lane.source_chunk_id is not None
                and lane.state in {"pending", "active"}
            ),
            None,
        )
        if target_index is None:
            return '<div class="manimux-chunk-spacer"></div>'
        target = self.lanes[target_index]
        source_index = self._lane_by_chunk.get(int(target.source_chunk_id))
        if source_index is None or source_index == target_index:
            return '<div class="manimux-chunk-spacer"></div>'
        source = self.lanes[source_index]
        if source.horizon_steps <= 0 or target.horizon_steps <= 0:
            return '<div class="manimux-chunk-spacer"></div>'

        source_start = min(
            source.horizon_steps,
            max(0, source.condition_from_index or source.cursor),
        )
        linked_steps = min(
            max(0, source.horizon_steps - source_start),
            max(0, target.overlap_steps),
        )
        if linked_steps <= 0:
            return '<div class="manimux-chunk-spacer"></div>'

        source_mid = 100.0 * (source_start + linked_steps / 2) / source.horizon_steps
        target_mid = 100.0 * (linked_steps / 2) / target.horizon_steps
        left = min(source_mid, target_mid)
        width = max(0.8, abs(source_mid - target_mid))
        source_edge = "top" if source_index == 0 else "bottom"
        target_edge = "top" if target_index == 0 else "bottom"
        return f"""
        <div class="manimux-chunk-handoff">
          <span class="manimux-chunk-link-v {source_edge}" style="left:{source_mid:.2f}%"></span>
          <span class="manimux-chunk-link-h" style="left:{left:.2f}%;width:{width:.2f}%"></span>
          <span class="manimux-chunk-link-v {target_edge} target"
                style="left:{target_mid:.2f}%"></span>
          <span class="manimux-chunk-link-label">condition · {linked_steps} steps</span>
        </div>
        """

    @staticmethod
    def _condition_range_html(lane: ChunkLane) -> str:
        if lane.horizon_steps <= 0:
            return ""
        ranges: list[str] = []
        if lane.condition_from_index is not None:
            start = min(lane.horizon_steps, max(0, lane.condition_from_index))
            count = lane.horizon_steps - start
            if count > 0:
                left = 100.0 * start / lane.horizon_steps
                width = 100.0 * count / lane.horizon_steps
                ranges.append(
                    '<span class="manimux-chunk-condition-range source" '
                    f'style="left:{left:.2f}%;width:{width:.2f}%" '
                    f'title="Condition source: {count} actions"></span>'
                )
        if (
            lane.conditioned
            and lane.source_chunk_id is not None
            and lane.condition_from_index is None
            and lane.state in {"pending", "active"}
        ):
            count = min(lane.horizon_steps, max(0, lane.overlap_steps))
            if count > 0:
                width = 100.0 * count / lane.horizon_steps
                ranges.append(
                    '<span class="manimux-chunk-condition-range target" '
                    f'style="left:0;width:{width:.2f}%" '
                    f'title="Conditioned prefix: {count} actions"></span>'
                )
        return "".join(ranges)

    def render_html(self) -> str:
        lane_html: list[str] = []
        for index, lane in enumerate(self.lanes):
            chunk = "—" if lane.chunk_id is None else f"#{lane.chunk_id}"
            lane_class = f"{html.escape(lane.state)} {'reverse' if index else ''}"
            cells = "".join(
                f'<span class="manimux-chunk-cell {self._cell_state(lane, cell)}'
                f'{" gripper-closed" if cell < len(lane.gripper_closed_steps) and lane.gripper_closed_steps[cell] else ""}"></span>'
                for cell in range(lane.horizon_steps)
            )
            condition_range = self._condition_range_html(lane)
            if not cells:
                cells = '<span class="manimux-chunk-empty">waiting for inference</span>'
            lane_html.append(
                f"""
                <div class="manimux-chunk-lane {lane_class}">
                  <div class="manimux-chunk-lane-head">
                    <strong>{'A' if index == 0 else 'B'} · {chunk}</strong>
                    <span>{html.escape(self._lane_summary(lane))}</span>
                  </div>
                  <div class="manimux-chunk-cells">{cells}{condition_range}</div>
                </div>
                """
            )
        runtime = html.escape(self.runtime.upper())
        lanes_with_handoff = (
            lane_html[0] + self._handoff_html() + lane_html[1]
        )
        return f"""
<style>
  div:has(> .manimux-chunk-anchor) {{
    position: fixed; left: 16px;
    top: calc(76px + var(--manimux-camera-height, 397px));
    width: var(--manimux-camera-width, clamp(300px, 26vw, 460px));
    z-index: 4; margin: 0; padding: 0 !important;
  }}
  .manimux-chunk-panel {{
    box-sizing: border-box; width: 100%; padding: 10px;
    border: 1px solid rgba(128, 138, 156, 0.35); border-radius: 12px;
    color: #eaf0fb; background: rgba(18, 23, 32, 0.94);
    box-shadow: 0 8px 28px rgba(0,0,0,0.16);
    font: 12px/1.35 system-ui, sans-serif;
  }}
  .manimux-chunk-title {{ display:flex; align-items:center; justify-content:space-between;
    margin: 0 1px 7px; color:#f7f9fc; }}
  .manimux-chunk-title span {{ color:#99a7bb; font-size:11px; letter-spacing:.08em; }}
  .manimux-chunk-lane {{ padding: 7px; border-radius: 8px; background:#0e131c; }}
  .manimux-chunk-lane.active {{ box-shadow: inset 0 0 0 1px rgba(110,91,255,.8); }}
  .manimux-chunk-lane.pending {{ box-shadow: inset 0 0 0 1px rgba(55,190,255,.55); }}
  .manimux-chunk-lane-head {{ display:flex; gap:8px; justify-content:space-between;
    margin-bottom:5px; color:#8f9db0; white-space:nowrap; overflow:hidden; }}
  .manimux-chunk-lane.reverse {{ display:flex; flex-direction:column; }}
  .manimux-chunk-lane.reverse .manimux-chunk-lane-head {{
    order:2; margin-top:5px; margin-bottom:0;
  }}
  .manimux-chunk-lane.reverse .manimux-chunk-cells {{ order:1; }}
  .manimux-chunk-lane-head strong {{ color:#f2f5fa; }}
  .manimux-chunk-lane-head span {{ overflow:hidden; text-overflow:ellipsis; }}
  .manimux-chunk-cells {{
    position:relative; display:flex; gap:1px; height:18px; min-width:0;
  }}
  .manimux-chunk-cell {{
    min-width:2px; flex:1 1 0; box-sizing:border-box;
    border:1px solid transparent; border-radius:2px; background:#303746;
  }}
  .manimux-chunk-cell.future {{
    border-color:#4b5565; background:transparent;
  }}
  .manimux-chunk-cell.executed {{ background:#7c3aed; }}
  .manimux-chunk-cell.current {{
    background:#a78bfa; box-shadow:0 0 7px rgba(167,139,250,.9);
  }}
  .manimux-chunk-cell.overlap {{
    background:repeating-linear-gradient(135deg,#606979 0 3px,#3d4553 3px 6px);
  }}
  .manimux-chunk-cell.frozen {{
    background:repeating-linear-gradient(135deg,#606979 0 3px,#3d4553 3px 6px);
  }}
  .manimux-chunk-cell.condition-source {{
    border-color:#4b5565; background:transparent;
  }}
  .manimux-chunk-cell.latency,
  .manimux-chunk-cell.latency-trimmed {{
    background:#f59e0b;
  }}
  .manimux-chunk-cell.latency-trimmed {{
    background:repeating-linear-gradient(135deg,#f59e0b 0 3px,#9a5d08 3px 6px);
  }}
  .manimux-chunk-cell.superseded,
  .manimux-chunk-cell.rejected {{
    background:repeating-linear-gradient(135deg,#303746 0 3px,#1d222c 3px 6px);
    opacity:.72;
  }}
  .manimux-chunk-cell.pending {{
    border-color:transparent;
    background:linear-gradient(90deg,#303746 20%,#8b5cf6 50%,#303746 80%);
    background-size:220% 100%; animation:manimux-chunk-shimmer 1.15s linear infinite;
  }}
  .manimux-chunk-cell.pending.overlap {{
    border-color:transparent;
    background:linear-gradient(90deg,#303746 20%,#8b5cf6 50%,#303746 80%);
    background-size:220% 100%;
  }}
  .manimux-chunk-cell.pending.frozen {{
    border-color:transparent;
    background:linear-gradient(90deg,#303746 20%,#8b5cf6 50%,#303746 80%);
    background-size:220% 100%;
  }}
  .manimux-chunk-cell.pending.latency {{
    border-color:transparent;
    background:linear-gradient(90deg,#9a5d08 20%,#f59e0b 50%,#9a5d08 80%);
    background-size:220% 100%;
  }}
  .manimux-chunk-cell.gripper-closed {{
    border-color:#ef4444; background:#ef4444;
  }}
  .manimux-chunk-cell.current.gripper-closed {{
    background:#f87171; box-shadow:0 0 7px rgba(248,113,113,.95);
  }}
  .manimux-chunk-cell.latency.gripper-closed {{
    background:#f59e0b; box-shadow:none;
  }}
  .manimux-chunk-cell.latency-trimmed.gripper-closed {{
    background:repeating-linear-gradient(135deg,#f59e0b 0 3px,#9a5d08 3px 6px);
    box-shadow:none;
  }}
  .manimux-chunk-condition-range {{
    position:absolute; top:-3px; bottom:-3px; z-index:3;
    box-sizing:border-box; border:2px solid #94a3b8; border-radius:5px;
    pointer-events:none;
  }}
  .manimux-chunk-empty {{ display:flex; align-items:center; padding-left:6px; width:100%;
    border:1px dashed #394354; border-radius:4px; color:#657185; }}
  .manimux-chunk-spacer {{ height:12px; }}
  .manimux-chunk-handoff {{
    position:relative; height:30px; margin:0 7px; color:#aeb8c8;
  }}
  .manimux-chunk-link-v {{
    position:absolute; width:0; height:calc(50% + 7px);
    border-left:2px solid #687386; z-index:2;
  }}
  .manimux-chunk-link-v.top {{ top:-7px; }}
  .manimux-chunk-link-v.bottom {{ bottom:-7px; }}
  .manimux-chunk-link-v.target::after {{
    content:""; position:absolute; left:-16px; width:32px; height:0;
    border-top:2px solid #687386;
  }}
  .manimux-chunk-link-v.target.top::after {{ top:0; }}
  .manimux-chunk-link-v.target.bottom::after {{ bottom:0; }}
  .manimux-chunk-link-h {{
    position:absolute; top:50%; height:0; border-top:2px solid #687386;
  }}
  .manimux-chunk-link-label {{
    position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
    padding:1px 6px; border-radius:8px; background:#121720;
    color:#9ba7b9; font-size:9px; white-space:nowrap;
  }}
  .manimux-chunk-legend {{ display:flex; gap:10px; margin-top:7px; color:#8794a7; font-size:10px; }}
  .manimux-chunk-legend i {{
    display:inline-block; width:8px; height:8px; margin-right:3px; border-radius:2px;
  }}
  @keyframes manimux-chunk-shimmer {{ to {{ background-position:-220% 0; }} }}
  @media (max-width: 900px) {{ div:has(> .manimux-chunk-anchor) {{ display:none; }} }}
</style>
<div class="manimux-chunk-anchor"></div>
<section class="manimux-chunk-panel">
  <div class="manimux-chunk-title"><strong>Action chunks</strong><span>{runtime}</span></div>
  {lanes_with_handoff}
  <div class="manimux-chunk-legend">
    <span><i style="background:#7c3aed"></i>executed</span>
    <span><i style="background:#a78bfa;box-shadow:0 0 5px rgba(167,139,250,.9)"></i>current</span>
    <span><i style="background:#f59e0b"></i>latency</span>
    <span><i style="background:#ef4444"></i>gripper closing / closed</span>
    <span><i style="border:2px solid #94a3b8;background:transparent"></i>condition</span>
    <span><i style="border:1px solid #4b5565"></i>future</span>
  </div>
</section>
"""
