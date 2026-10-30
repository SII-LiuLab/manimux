"""Hosted Dream-1 API behind the existing single-inflight PolicyModel worker."""

import os

import numpy as np

from manimux.policies.base import action_interval
from manimux.policies.capabilities import PolicyCapabilities, metadata_mismatches
from manimux.policies.dream.codec import CAMERAS, DreamCodec
from manimux.policies.response_log import capture_response


class DreamPolicyModel:
    def __init__(self, config):
        self.config = config
        options = config["options"]
        allowed = {
            "checkpoint",
            "api_key_env",
            "request_timeout_s",
            "obs_window",
            "image_size",
            "jpeg_quality",
            "response_log_dir",
        }
        if set(options) - allowed:
            raise ValueError(
                f"Unsupported Dream API option names: {sorted(set(options) - allowed)}"
            )
        self.checkpoint = options["checkpoint"]
        if self.checkpoint not in {"dreamlabs/dream-1-beta", "dreamlabs/dream-1-beta-flash"}:
            raise ValueError("This adapter supports the Dream-1 beta YAM base models only")
        if type(options.get("obs_window", 1)) is not int or options.get("obs_window", 1) != 1:
            raise ValueError("Dream API currently supports one current frame per camera only")
        self.timeout = float(options.get("request_timeout_s", config["timeout_s"]))
        self.dt = float(action_interval(config))
        if not np.isfinite([self.timeout, self.dt]).all() or min(self.timeout, self.dt) <= 0:
            raise ValueError(
                "Dream request timeout and action interval must be finite and positive"
            )
        self.key_env = options.get("api_key_env", "DREAM_API_KEY")
        if not isinstance(self.key_env, str) or not self.key_env.isidentifier():
            raise ValueError("api_key_env must name an environment variable")
        self.codec = DreamCodec(config)
        self.client = None
        self.session_id = None
        self._metadata = {}
        self._instruction = None

    def reset(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Dream reset requires a non-empty session ID")
        self.close()
        key = os.environ.get(self.key_env)
        if not key:
            raise ValueError(
                f"Set {self.key_env} in the runtime environment; never put keys in YAML"
            )
        try:
            from dream import DreamClient
        except ImportError:
            raise RuntimeError(
                "Install the hosted API client with pip install 'manimux[dream]'"
            ) from None
        client = DreamClient(api_key=key)
        try:
            listed = client.get_models()
            model = next((item for item in listed if item.id == self.checkpoint), None)
            if model is None or not model.servable:
                raise ValueError("Dream checkpoint is not available to this API account")
            if model.embodiment != "yam" or set(model.cameras or []) != CAMERAS:
                raise ValueError("Dream checkpoint does not match the YAM three-camera contract")
            self._metadata = {
                "server": "dream_api",
                "model": {"checkpoint": model.id, "embodiment": model.embodiment},
            }
            expected = self.config.get("expected_backend")
            if expected and (errors := metadata_mismatches(expected, self._metadata)):
                raise ValueError(f"Dream backend identity mismatch: {errors}")
            client.policy(
                {
                    "checkpoint": self.checkpoint,
                    "robot": "yam",
                    "obs_window": 1,
                    "timeout_ms": max(1, round(self.timeout * 1000)),
                    "action_hz": 1 / self.dt,
                    # Test/runtime observations go only to inference, not the SDK's
                    # additional background episode-recording service.
                    "_record": False,
                }
            )
        except Exception as exc:
            client.close()
            self._metadata = {}
            raise RuntimeError(
                f"Dream startup failed: {str(exc).replace(key, '[REDACTED]')}"
            ) from None
        self.client = client
        self.session_id = session_id
        self._instruction = None

    def infer(self, request):
        if self.client is None or request.session_id != self.session_id:
            raise ValueError("Dream request does not belong to the active session")
        from dream import DreamSessionExpired

        frames, observation = self.codec.encode(request)
        key = os.environ.get(self.key_env, "")
        try:
            if self._instruction is not None and request.instruction != self._instruction:
                # A new instruction (an RPent subtask) is a new Dream episode. It also drops a
                # session the gateway may have evicted while the agent was planning.
                self.client.reset_episode()
            self._instruction = request.instruction
            self.client.set_observation(
                frames, observation, timestamp=request.observation_time_ns / 1e9
            )
            try:
                raw = self.client.get_action()
            except DreamSessionExpired:
                # The gateway replaced an idle session and already adopted the new one, but
                # returned no action; ask once on it, as the SDK documents.
                raw = self.client.get_action()
        except Exception as exc:
            # No unbounded retry: pool warming, billing, auth and timeouts return
            # to the runtime as failures; do not consume another observation's deadline.
            message = str(exc)
            if key:
                message = message.replace(key, "[REDACTED]")
            raise RuntimeError(
                f"Dream inference failed ({type(exc).__name__}): {message}"
            ) from None
        capture_response(raw, metadata={"timing": self.client.last_timing})
        result = self.codec.decode(raw)
        result["sampling_stats"]["dream_api"].update(self.client.last_timing or {})
        return result

    def capabilities(self):
        return PolicyCapabilities(
            sampling_modes=frozenset({"default"}),
            multi_inflight=False,
            backend_metadata=dict(self._metadata),
        )

    def close(self):
        client, self.client = self.client, None
        self.session_id = None
        self._metadata = {}
        if client is not None:
            client.close()


def build_model(config):
    return DreamPolicyModel(config)
