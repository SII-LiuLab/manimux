from __future__ import annotations

import time

import numpy as np

from manimux.config import load_config
from manimux.integrations.sapolicy_yam.policy_plugin import (
    SAPolicyXPolicyRequest,
    build_adapter,
)
from manimux.types import ObservationSnapshot, RobotState, SensorFrame

CONFIG = "configs/sapolicy/yam/infra/manimux-xpl.yaml"


def _snapshot(now_ns: int) -> ObservationSnapshot:
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
                "right_arm": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            },
            monotonic_ns=now_ns,
            sequence=1,
        ),
        frames={
            name: SensorFrame(
                name=name,
                data=np.zeros((48, 64, 3), dtype=np.uint8),
                capture_monotonic_ns=now_ns,
                sequence=1,
            )
            for name in ("front_camera", "left_camera", "right_camera")
        },
    )


def test_sapolicy_xpl_infra_uses_ws_transport() -> None:
    config = load_config(CONFIG)
    assert config.policy.worker == "xpolicylab_ws"
    assert config.policy.adapter == "sapolicy_yam"
    assert config.policy.horizon_steps == 16


def test_sapolicy_adapter_prepares_xpolicylab_additional_info() -> None:
    config = load_config(CONFIG)
    adapter = build_adapter(config.robot, config.policy)
    now_ns = time.monotonic_ns()
    from manimux.types import InferenceRequest

    prepared = adapter.prepare_request(
        InferenceRequest(
            session_id="session",
            request_seq=1,
            observation_time_ns=now_ns,
            deadline_ns=now_ns + 1_000_000_000,
            observation=_snapshot(now_ns),
            instruction="put bottles in bin",
        )
    )

    assert isinstance(prepared, SAPolicyXPolicyRequest)
    sap = prepared.xpolicylab_additional_info["sapolicy"]
    assert np.asarray(sap["left_endpose"]).shape == (7,)
    assert np.asarray(sap["right_endpose"]).shape == (7,)
    assert np.asarray(sap["intrinsics"]["top"]).shape == (3, 3)
    assert set(sap["intrinsics"]) == {"top", "left", "right"}
    # Wire RGB is stretch-resized; K stays at the native calibration.
    assert sap["image_native_hw"] == {
        "top": [48, 64],
        "left": [48, 64],
        "right": [48, 64],
    }
    for name in ("front_camera", "left_camera", "right_camera"):
        assert prepared.observation.frames[name].data.shape == (168, 224, 3)
        assert prepared.observation.frames[name].data.dtype == np.uint8
