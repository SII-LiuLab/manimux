"""Existing MOLMOACT HTTP policy client; model implementation remains separate."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import InferenceRequest

DEFAULT_SERVER = "http://127.0.0.1:8202"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_CAMERA_MAP = {
    "left_cam": "left_camera",
    "top_cam": "front_camera",
    "right_cam": "right_camera",
}


def _server_url(server: str) -> str:
    normalized = server.strip().rstrip("/")
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    if not normalized.endswith("/act"):
        normalized += "/act"
    return normalized


class MolmoActHttpPolicyModel:
    """HTTP inference backend; robot semantics stay in ``MolmoActYamAdapter``."""

    def __init__(self, config: dict) -> None:
        self._url = _server_url(config["options"].get("server", DEFAULT_SERVER))
        self._health_url = self._url.removesuffix("/act") + "/healthz"
        self._group_order = tuple(config["adapter"].get("group_order", DEFAULT_GROUP_ORDER))
        self._camera_map = dict(DEFAULT_CAMERA_MAP)
        camera_map = config["adapter"].get("camera_map")
        if camera_map is not None:
            self._camera_map = dict(camera_map)
        self._normalization_tag = config["options"].get("normalization_tag", "yam_dual_molmoact2")
        self._enable_cuda_graph = bool(config["options"].get("enable_cuda_graph", True))
        num_steps = config["options"].get("num_steps")
        if num_steps is not None and (not isinstance(num_steps, int) or num_steps <= 0):
            raise ValueError("policy.options.num_steps must be a positive integer")
        self._num_steps = num_steps
        self._timeout_s = float(config["options"].get("http_timeout_s", config["timeout_s"]))
        if self._timeout_s <= 0:
            raise ValueError("policy.options.http_timeout_s must be positive")
        self._session_id: str | None = None

    def reset(self, session_id: str) -> None:
        import requests

        response = requests.get(self._health_url, timeout=self._timeout_s)
        if response.status_code != 200:
            raise RuntimeError(f"MolmoAct health check failed with status {response.status_code}")
        payload = response.json()
        if payload.get("status") != "ok":
            raise RuntimeError(f"MolmoAct health check is not ready: {payload!r}")
        self._session_id = session_id

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id:
            raise RuntimeError("MolmoAct session is not initialized")
        snapshot = request.observation
        missing_groups = [name for name in self._group_order if name not in snapshot.state.groups]
        if missing_groups:
            raise ValueError(f"MolmoAct observation is missing groups: {missing_groups}")
        missing_frames = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing_frames:
            raise ValueError(f"MolmoAct observation is missing cameras: {missing_frames}")

        payload: dict[str, Any] = {
            wire_name: snapshot.frames[source_name].data
            for wire_name, source_name in self._camera_map.items()
        }
        payload.update(
            {
                "timestamp": time.time(),
                "instruction": request.instruction,
                "state": np.concatenate(
                    [snapshot.state.groups[name] for name in self._group_order]
                ),
                "normalization_tag": self._normalization_tag,
                "enable_cuda_graph": self._enable_cuda_graph,
            }
        )
        if self._num_steps is not None:
            payload["num_steps"] = self._num_steps

        # An RTC runtime sends the inpainting condition on a request subclass;
        # the default runtime never sets it and the payload is unchanged.
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if condition is not None and weights is not None:
            payload["action_condition"] = np.asarray(condition, dtype=np.float32)
            payload["action_condition_weights"] = np.asarray(weights, dtype=np.float32)
            payload["rtc_beta"] = float(getattr(request, "rtc_beta", 5.0))

        import json_numpy
        import requests

        remaining_s = max(0.001, (request.deadline_ns - time.monotonic_ns()) / 1e9)
        response = requests.post(
            self._url,
            headers={"Content-Type": "application/json"},
            data=json_numpy.dumps(payload),
            timeout=min(self._timeout_s, remaining_s),
        )
        if response.status_code != 200:
            raise RuntimeError(f"MolmoAct server error {response.status_code}: {response.text}")
        decoded = json_numpy.loads(response.text)
        actions = np.asarray(decoded["actions"], dtype=np.float64)
        if actions.ndim != 2 or not actions.shape[0] or not np.isfinite(actions).all():
            raise ValueError("MolmoAct actions must be a non-empty finite matrix")
        return np.ascontiguousarray(actions)

    def close(self) -> None:
        self._session_id = None

    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(sampling_modes=frozenset({"default", "rtc"}))


def build_model(config: dict) -> MolmoActHttpPolicyModel:
    return MolmoActHttpPolicyModel(config)
