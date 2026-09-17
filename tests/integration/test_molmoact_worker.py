from __future__ import annotations

import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import json_numpy
import numpy as np
import pytest

from manimux.cli import load_config
from manimux.policies import build_policy_adapter
from manimux.policies.worker import PolicyWorkerClient
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


class _MolmoHandler(BaseHTTPRequestHandler):
    received: queue.Queue[dict[str, Any]] = queue.Queue()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        size = int(self.headers["Content-Length"])
        payload = json_numpy.loads(self.rfile.read(size).decode("utf-8"))
        self.received.put(payload)
        actions = np.arange(30 * 14, dtype=np.float32).reshape(30, 14)
        body = json_numpy.dumps({"actions": actions}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


class _UnhealthyMolmoHandler(_MolmoHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = b'{"status":"starting"}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_molmoact_http_worker_round_trip_to_canonical_action_chunk() -> None:
    _MolmoHandler.received = queue.Queue()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MolmoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    config["policy"]["options"]["server"] = f"http://127.0.0.1:{server.server_port}"
    session_id = "molmoact-test-session"
    worker = PolicyWorkerClient(config["policy"], session_id)
    now_ns = time.monotonic_ns()
    frames = {
        name: SensorFrame(
            name=name,
            data=np.full((8, 8, 3), index, dtype=np.uint8),
            capture_monotonic_ns=now_ns,
            sequence=1,
        )
        for index, name in enumerate(("left_camera", "front_camera", "right_camera"), 1)
    }
    request = InferenceRequest(
        session_id=session_id,
        request_seq=3,
        observation_time_ns=now_ns,
        deadline_ns=now_ns + 2_000_000_000,
        observation=ObservationSnapshot(
            state=RobotState(
                groups={"left_arm": np.arange(7), "right_arm": np.arange(7, 14)},
                monotonic_ns=now_ns,
                sequence=1,
            ),
            frames=frames,
        ),
        instruction="test instruction",
    )

    try:
        worker.start()
        worker.submit_latest(request)
        deadline = time.monotonic() + 5.0
        response = None
        while response is None and time.monotonic() < deadline:
            response = worker.poll()
            time.sleep(0.01)
        assert response is not None
        assert response.error is None
        assert response.raw_action is not None

        adapter = build_policy_adapter(config["robot"], config["policy"])
        chunk = adapter.decode_action(
            response.raw_action,
            ActionContext(
                request_seq=response.request_seq,
                observation_time_ns=response.observation_time_ns,
                created_time_ns=response.finished_time_ns,
            ),
        )
        assert chunk.groups["left_arm"].shape == (30, 7)
        assert chunk.groups["right_arm"].shape == (30, 7)

        payload = _MolmoHandler.received.get(timeout=1.0)
        assert payload["instruction"] == "test instruction"
        np.testing.assert_array_equal(payload["state"], np.arange(14))
        np.testing.assert_array_equal(payload["left_cam"], frames["left_camera"].data)
        np.testing.assert_array_equal(payload["top_cam"], frames["front_camera"].data)
        np.testing.assert_array_equal(payload["right_cam"], frames["right_camera"].data)
    finally:
        worker.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_molmoact_worker_fails_startup_before_hardware_when_server_is_unhealthy() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UnhealthyMolmoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    config["policy"]["options"]["server"] = f"http://127.0.0.1:{server.server_port}"
    worker = PolicyWorkerClient(config["policy"], "unhealthy-session")

    try:
        with pytest.raises(RuntimeError, match="health check failed"):
            worker.start()
        assert not worker.is_alive
    finally:
        worker.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
