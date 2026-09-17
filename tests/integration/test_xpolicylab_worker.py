"""End-to-end tests against a stand-in XPolicyLab WebSocket policy server.

The fake server implements XPolicyLab's frame protocol rather than importing
its code, which is the same relationship the real bridge has: if these pass, we
speak the wire correctly without ever depending on the package.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

pytest.importorskip("msgpack")
pytest.importorskip("msgpack_numpy")
pytest.importorskip("websockets")

from manimux.integrations.xpolicylab.ws_client import (  # noqa: E402
    XPolicyLabProtocolError,
    pack_frame,
    unpack_frame,
)

HORIZON = 30


class FakeXPolicyLabServer:
    """Answers hello/reset/infer frames the way XPolicyLab's server does."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._fail_on = fail_on
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.port = 0

    def start(self) -> None:
        from websockets.sync.server import serve

        self._server = serve(self._handle, "127.0.0.1", 0, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def _handle(self, websocket: Any) -> None:
        for message in websocket:
            frame = unpack_frame(message)
            reply = self._reply_to(frame)
            if reply is not None:
                websocket.send(pack_frame(reply))

    def _reply_to(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        kind = frame["message_type"]
        payload = frame.get("payload") or {}
        envelope = {
            "message_id": frame["message_id"],
            "evaluation_id": frame["evaluation_id"],
            "trial_id": frame.get("trial_id"),
            "action_case_id": None,
            "repeat_index": None,
            "step": frame.get("step", 0),
            "sent_at": "1970-01-01T00:00:00+00:00",
        }
        if kind == "hello":
            self.calls.put(("hello", None))
            return {
                **envelope,
                "message_type": "hello_ack",
                "payload": {"capabilities": {"sampling_modes": ["default", "rtc"]}},
            }
        if kind == "reset":
            self.calls.put(("reset", None))
            return {**envelope, "message_type": "reset_result", "payload": {"result": None}}
        if kind == "infer":
            self.calls.put(("infer", payload))
            if self._fail_on == "infer":
                return {
                    **envelope,
                    "message_type": "error",
                    "payload": {"message": "infer exploded"},
                }
            return {
                **envelope,
                "message_type": "infer_result",
                "payload": {"actions": _action_steps(), "latency_ms": 1.0},
            }
        raise AssertionError(f"unexpected frame {kind!r}")


def _action_steps() -> list[dict[str, np.ndarray]]:
    return [
        {
            "left_arm_joint_state": np.full(6, step * 0.01, dtype=np.float32),
            "left_ee_joint_state": np.array([0.5], dtype=np.float32),
            "right_arm_joint_state": np.full(6, -step * 0.01, dtype=np.float32),
            "right_ee_joint_state": np.array([0.25], dtype=np.float32),
        }
        for step in range(HORIZON)
    ]


def _snapshot() -> ObservationSnapshot:
    rng = np.random.default_rng(0)
    frames = {
        name: SensorFrame(
            name=name,
            data=rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8),
            capture_monotonic_ns=1,
            sequence=1,
        )
        for name in ("left_camera", "front_camera", "right_camera")
    }
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.arange(7, dtype=np.float64),
                "right_arm": np.arange(7, dtype=np.float64) + 10.0,
            },
            monotonic_ns=1,
            sequence=1,
        ),
        frames=frames,
    )


def _request(session_id: str) -> InferenceRequest:
    return InferenceRequest(
        session_id=session_id,
        request_seq=3,
        observation_time_ns=100,
        deadline_ns=10**18,
        observation=_snapshot(),
        instruction="pick the red ball up",
    )


@pytest.fixture
def server() -> Iterator[FakeXPolicyLabServer]:
    fake = FakeXPolicyLabServer()
    fake.start()
    yield fake
    fake.stop()


def _configs(server_url: str) -> tuple[Any, Any]:
    config = load_config("configs/xpolicylab/yam/infra/smoke.yaml")
    config["policy"]["options"]["server"] = server_url
    return config["robot"], config["policy"]


def test_a_full_round_trip_produces_a_canonical_chunk(server: FakeXPolicyLabServer) -> None:
    robot, policy = _configs(server.url)
    model = build_policy_model(policy)
    adapter = build_policy_adapter(robot, policy)

    model.reset("session-1")
    raw = model.infer(_request("session-1"))
    model.close()

    chunk = adapter.decode_action(
        raw, ActionContext(request_seq=3, observation_time_ns=100, created_time_ns=200)
    )
    assert chunk.action_space == "joint_position"
    assert chunk.horizon_steps == HORIZON
    assert chunk.groups["left_arm"].shape == (HORIZON, 7)
    assert chunk.groups["right_arm"].shape == (HORIZON, 7)
    assert chunk.dt_ns == 33_333_333
    np.testing.assert_allclose(chunk.groups["left_arm"][2][:6], 0.02, atol=1e-6)
    np.testing.assert_allclose(chunk.groups["left_arm"][2][6], 0.5, atol=1e-6)


def test_the_handshake_and_call_order_match_the_protocol(server: FakeXPolicyLabServer) -> None:
    _, policy = _configs(server.url)
    model = build_policy_model(policy)
    model.reset("session-2")
    model.infer(_request("session-2"))
    model.close()

    order = []
    while not server.calls.empty():
        order.append(server.calls.get()[0])
    assert order == ["hello", "reset", "infer"]


def test_handshake_reports_sampling_capabilities(server: FakeXPolicyLabServer) -> None:
    _, policy = _configs(server.url)
    model = build_policy_model(policy)
    model.reset("session-capabilities")

    assert model.capabilities().sampling_modes == frozenset({"default", "rtc"})
    model.close()


def test_the_observation_survives_the_wire_intact(server: FakeXPolicyLabServer) -> None:
    _, policy = _configs(server.url)
    snapshot = _snapshot()
    request = InferenceRequest(
        session_id="session-3",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=10**18,
        observation=snapshot,
        instruction="stack the bowls",
    )

    model = build_policy_model(policy)
    model.reset("session-3")
    model.infer(request)
    model.close()

    received = {}
    while not server.calls.empty():
        name, obs = server.calls.get()
        if name == "infer":
            received = obs["observation"]

    assert received["data_format_version"] == "v1.0"
    assert received["instruction"] == "stack the bowls"
    # The rate is declared by us, not chosen by the model.
    assert received["additional_info"]["frequency"] == pytest.approx(30.0)

    state = received["state"]
    np.testing.assert_allclose(state["left_arm_joint_state"], np.arange(6))
    np.testing.assert_allclose(state["left_ee_joint_state"], [6.0])
    np.testing.assert_allclose(state["right_ee_joint_state"], [16.0])

    # Images must be byte-identical after the msgpack-numpy round trip.
    assert set(received["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    np.testing.assert_array_equal(
        received["vision"]["cam_left_wrist"]["color"], snapshot.frames["left_camera"].data
    )
    np.testing.assert_array_equal(
        received["vision"]["cam_head"]["color"], snapshot.frames["front_camera"].data
    )
    assert received["vision"]["cam_head"]["shape"] == [48, 64]


def test_a_server_error_frame_raises_rather_than_returning_a_chunk() -> None:
    fake = FakeXPolicyLabServer(fail_on="infer")
    fake.start()
    try:
        _, policy = _configs(fake.url)
        model = build_policy_model(policy)
        model.reset("session-4")
        with pytest.raises(XPolicyLabProtocolError, match="infer exploded"):
            model.infer(_request("session-4"))
        model.close()
    finally:
        fake.stop()


def test_an_rtc_request_forwards_the_condition_and_soft_mask(
    server: FakeXPolicyLabServer,
) -> None:
    from manimux.runtime.rtc.request import RtcInferenceRequest

    _, policy = _configs(server.url)
    model = build_policy_model(policy)
    model.reset("session-5")
    request = RtcInferenceRequest(
        session_id="session-5",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=10**18,
        observation=_snapshot(),
        action_condition=np.zeros((HORIZON, 14)),
        condition_weights=np.ones(HORIZON),
    )
    model.infer(request)
    model.close()

    calls = []
    while not server.calls.empty():
        calls.append(server.calls.get())
    assert [name for name, _ in calls] == ["hello", "reset", "infer"]
    rtc_payload = calls[-1][1]["sampling"]
    assert rtc_payload["mode"] == "rtc"
    np.testing.assert_array_equal(rtc_payload["action_condition"], np.zeros((HORIZON, 14)))
    np.testing.assert_array_equal(rtc_payload["condition_weights"], np.ones(HORIZON))
    assert rtc_payload["beta"] == pytest.approx(5.0)


def test_infer_after_close_does_not_reach_the_server(server: FakeXPolicyLabServer) -> None:
    _, policy = _configs(server.url)
    model = build_policy_model(policy)
    model.reset("session-6")
    model.close()
    with pytest.raises(RuntimeError, match="session is not initialized"):
        model.infer(_request("session-6"))
