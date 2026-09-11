from __future__ import annotations

from manimux.types import ActionChunk, ActionHorizon, GroupVector


def _vector_payload(groups: GroupVector | None) -> dict[str, list[float]] | None:
    if groups is None:
        return None
    return {name: values.tolist() for name, values in groups.items()}


def build_plan_boundary_payload(
    *,
    step: int,
    monotonic_ns: int,
    blend_anchor_source: str,
    blend_steps: int,
    trimmed_steps: int,
    previous_reference: GroupVector | None,
    previous_command: GroupVector,
    last_command: GroupVector,
    measured: GroupVector,
    chunk: ActionChunk,
    committed: ActionHorizon,
) -> dict[str, object]:
    raw_start = trimmed_steps
    return {
        "step": step,
        "monotonic_ns": monotonic_ns,
        "plan_id": chunk.plan_id,
        "request_seq": chunk.request_seq,
        "action_space": chunk.action_space,
        "blend_anchor_source": blend_anchor_source,
        "blend_steps": blend_steps,
        "trimmed_steps": trimmed_steps,
        "previous_reference": _vector_payload(previous_reference),
        "previous_command": _vector_payload(previous_command),
        "last_command": _vector_payload(last_command),
        "measured": _vector_payload(measured),
        "raw_first": {
            name: values[raw_start].tolist() for name, values in chunk.groups.items()
        },
        "committed_first": {name: values[0].tolist() for name, values in committed.groups.items()},
    }
