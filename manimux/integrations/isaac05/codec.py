"""Policy-independent wire contract for the public Isaac 0.5 LIBERO checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Isaac05BaseContract:
    camera_roles: tuple[str, str] = ("primary", "wrist")
    proprio_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    model_chunk_size: int = 50
    target_fps: float = 20.0
    action_semantics: str = "checkpoint_native_absolute_ee"


def _rgb_image(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Isaac camera {name!r} must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError(f"Isaac camera {name!r} must use uint8 pixels")
    return np.ascontiguousarray(image)


def build_wire_observation(
    *,
    primary_image: Any,
    wrist_image: Any,
    state: Any,
    instruction: str,
    timestep: int = 0,
) -> dict[str, Any]:
    """Build the XPolicy wire payload consumed by the Isaac 0.5 wrapper."""

    contract = Isaac05BaseContract()
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_array.shape != (contract.proprio_dim,) or not np.isfinite(state_array).all():
        raise ValueError(
            "Isaac state must contain "
            f"{contract.proprio_dim} finite values, got {state_array.shape}"
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Isaac instruction must be a non-empty string")
    if not isinstance(timestep, int) or timestep < 0:
        raise ValueError("Isaac timestep must be a non-negative integer")
    return {
        "vision": {
            "primary": {"color": _rgb_image(primary_image, "primary")},
            "wrist": {"color": _rgb_image(wrist_image, "wrist")},
        },
        "state": {"observation.state": np.ascontiguousarray(state_array)},
        "instruction": instruction.strip(),
        "additional_info": {
            "frequency": contract.target_fps,
            "timestep": timestep,
        },
        "data_format_version": "v1.0",
        "env_idx": 0,
    }


def decode_wire_actions(raw: object) -> np.ndarray:
    """Decode the public checkpoint's XPolicy result without changing EE semantics."""

    if isinstance(raw, Mapping) and "actions" in raw:
        raw = raw["actions"]
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("Isaac actions must be a sequence of per-step mappings")
    rows: list[np.ndarray] = []
    for index, step in enumerate(raw):
        if not isinstance(step, Mapping) or "action" not in step:
            raise ValueError(f"Isaac action step {index} must contain an 'action' vector")
        row = np.asarray(step["action"], dtype=np.float64).reshape(-1)
        rows.append(row)
    contract = Isaac05BaseContract()
    array = np.asarray(rows, dtype=np.float64)
    expected = (contract.action_horizon, contract.action_dim)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"Isaac actions must be finite with shape {expected}, got {array.shape}")
    return np.ascontiguousarray(array)
