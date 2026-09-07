from __future__ import annotations

import numpy as np
import pytest

from manimux.integrations.isaac05 import build_wire_observation, decode_wire_actions


def test_build_wire_observation_preserves_official_base_contract() -> None:
    primary = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.ones((256, 256, 3), dtype=np.uint8)
    observation = build_wire_observation(
        primary_image=primary,
        wrist_image=wrist,
        state=np.arange(8, dtype=np.float32),
        instruction="Pick up the object.",
        timestep=16,
    )
    assert set(observation["vision"]) == {"primary", "wrist"}
    assert observation["state"]["observation.state"].shape == (8,)
    assert observation["additional_info"] == {"frequency": 20.0, "timestep": 16}


def test_build_wire_observation_rejects_non_libero_state_width() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="8 finite values"):
        build_wire_observation(
            primary_image=image,
            wrist_image=image,
            state=np.zeros(14),
            instruction="Pick up the object.",
        )


def test_decode_wire_actions_preserves_native_ee_vectors() -> None:
    expected = np.arange(56, dtype=np.float64).reshape(8, 7)
    raw = [{"action": row.astype(np.float32)} for row in expected]
    actual = decode_wire_actions(raw)
    np.testing.assert_allclose(actual, expected)


def test_decode_wire_actions_rejects_full_model_horizon() -> None:
    raw = [{"action": np.zeros(7)} for _ in range(50)]
    with pytest.raises(ValueError, match=r"\(8, 7\)"):
        decode_wire_actions(raw)
