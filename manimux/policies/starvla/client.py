"""PolicyModel client for StarVLA's independent native WebSocket service."""

import time
import uuid

import numpy as np

from manimux.policies.capabilities import PolicyCapabilities, metadata_mismatches
from manimux.policies.starvla.codec import StarVlaCodec
from manimux.policies.starvla.wire import pack, unpack


class StarVlaPolicyModel:
    """Own one connection and episode; model computation stays in the server."""

    def __init__(self, config):
        self.config = config
        options = config["options"]
        self.url = options["server"]
        self.timeout = float(options.get("request_timeout_s", config["timeout_s"]))
        self.connect_timeout = float(config["startup_timeout_s"])
        if (
            not np.isfinite([self.timeout, self.connect_timeout]).all()
            or min(self.timeout, self.connect_timeout) <= 0
        ):
            raise ValueError("StarVLA timeouts must be positive and finite")
        self.connection = None
        self.session_id = None
        self.contract = None
        self.codec = None
        self._aac_previous = None
        self._aac_kinematics = None
        self._aac_robot_config = None

    def reset(self, session_id: str) -> None:
        """Reconnect, validate the deployment, and start a fresh server session."""
        from websockets.sync.client import connect

        self.close()
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                self.connection = connect(
                    self.url,
                    compression=None,
                    max_size=None,
                    open_timeout=max(0.01, deadline - time.monotonic()),
                    ping_interval=None,
                )
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"StarVLA service did not start at {self.url}") from None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        try:
            metadata = unpack(self.connection.recv(timeout=self.timeout))
            if (
                metadata.get("env") != "starvla_policy_server"
                or metadata.get("protocol_version") != 1
                or metadata.get("supports_reset") is not True
            ):
                raise ValueError("Server does not implement the required StarVLA serving contract")
            self.contract = metadata["serving_contract"]
            if self.contract["action_horizon"] != self.config["horizon_policy_steps"]:
                raise ValueError("StarVLA horizon differs from runtime configuration")
            if (
                self.contract["image_color_order"] != "rgb"
                or self.contract["action_output"] != "unnormalized_env"
            ):
                raise ValueError("Unsupported StarVLA image/action contract")
            expected = self.config.get("expected_backend")
            if expected:
                mismatch = metadata_mismatches(expected, self._backend_metadata())
                if mismatch:
                    raise ValueError(f"StarVLA backend identity mismatch: {mismatch}")
            self.codec = StarVlaCodec(self.config["adapter"], self.contract)
            reply = self._call("reset", session_id=session_id)
            if reply.get("session_id") != session_id:
                raise ValueError("StarVLA reset acknowledged another session")
            self.session_id = session_id
            self._aac_previous = None
        except Exception:
            self.close()
            raise

    def _call(self, kind, **fields):
        if self.connection is None:
            raise RuntimeError("StarVLA client is not connected")
        request_id = uuid.uuid4().hex
        try:
            self.connection.send(pack({"type": kind, "request_id": request_id, **fields}))
            result = unpack(self.connection.recv(timeout=self.timeout))
            if result.get("request_id") != request_id:
                raise ValueError("StarVLA response request_id mismatch")
            expected = "inference_result" if kind == "infer" else kind
            if result.get("type") != expected:
                raise ValueError("StarVLA response type mismatch")
            if type(result.get("ok")) is not bool:
                raise ValueError("StarVLA response ok must be a boolean")
        except Exception:
            self.close()
            raise
        # A valid error response completes this RPC without invalidating the
        # connection or episode. Let the worker reject only this observation.
        if not result["ok"]:
            raise RuntimeError(f"StarVLA request failed: {result.get('error')}")
        return result

    def infer(self, request):
        if self.session_id is None or request.session_id != self.session_id:
            raise RuntimeError("StarVLA inference session is not initialized")
        sampling = self._sampling(request)
        payload = self.codec.encode(request)
        result = self._call(
            "infer", session_id=self.session_id, payload={**payload, "sampling": sampling}
        )["data"]
        actions = np.asarray(result["actions"])
        count = sampling.get("num_samples", 1)
        expected = (count, self.contract["action_horizon"], self.contract["action_dim"])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"StarVLA returned invalid actions; expected {expected}")
        if sampling["mode"] == "aac":
            return self._select_aac(request, actions)
        decoded = self.codec.decode(actions[0])
        mode = sampling["mode"]
        if mode in {"paint", "autohorizon"}:
            if mode not in result:
                raise ValueError(f"StarVLA omitted {mode} sampler metadata")
            decoded[mode] = result[mode]
        return decoded

    def _sampling(self, request):
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if (condition is None) != (weights is None):
            raise ValueError("RTC condition and weights must be supplied together")
        prefix = getattr(request, "paint_action_prefix", None)
        delay = getattr(request, "paint_delay_steps", None)
        if (prefix is None) != (delay is None):
            raise ValueError("PAINT prefix and delay must be supplied together")
        count = getattr(request, "aac_num_samples", None)
        modes = [
            m
            for m, used in [
                ("rtc", condition is not None),
                ("paint", prefix is not None),
                ("aac", count is not None),
                ("autohorizon", getattr(request, "autohorizon", False)),
            ]
            if used
        ]
        if len(modes) > 1:
            raise ValueError("Cannot combine specialized sampling modes")
        mode = modes[0] if modes else "default"
        if mode not in self.contract["sampling_modes"]:
            raise ValueError(f"StarVLA does not support {mode} for this deployment")
        if mode == "rtc":
            return dict(
                mode=mode,
                action_condition=self.codec.native_condition(condition),
                condition_weights=np.asarray(weights),
                beta=float(request.rtc_beta),
            )
        if mode == "paint":
            return dict(
                mode=mode, action_prefix=self.codec.native_condition(prefix), delay_steps=delay
            )
        if mode == "aac":
            return dict(mode=mode, num_samples=count)
        return {"mode": mode}

    def _select_aac(self, request, actions):
        from manimux.embodiments.robot.base import RobotModel
        from manimux.policies.aac import load_ee_action_stats, select_ee_chunk

        source = request.aac_robot_config
        if not source:
            raise ValueError("AAC requires the configured offline robot geometry")
        if self._aac_robot_config != source:
            self._aac_kinematics = RobotModel.from_config(source).kinematics
            self._aac_robot_config = source
        candidates = [self.codec.decode(rows)["actions"] for rows in actions]
        stats = load_ee_action_stats(request.aac_ee_stats_path, layouts=self.codec.layouts)
        groups, selection, self._aac_previous = select_ee_chunk(
            candidates,
            layouts=self.codec.layouts,
            current_groups=request.observation.state.groups,
            kinematics=self._aac_kinematics,
            ee_stats=stats,
            motion_threshold=request.aac_motion_threshold,
            chunk_id_selector=request.aac_chunk_id_selector,
            previous=self._aac_previous,
            backward_beta=request.aac_backward_beta,
        )
        return {
            "format": "joint",
            "actions": groups,
            "aac": selection.metadata(),
            "action_semantics": self.contract["action_semantics"],
        }

    def _backend_metadata(self):
        return {"server": "starvla_policy_server", "model": dict(self.contract or {})}

    def capabilities(self) -> PolicyCapabilities:
        modes = frozenset(self.contract["sampling_modes"]) if self.contract else frozenset()
        return PolicyCapabilities(sampling_modes=modes, backend_metadata=self._backend_metadata())

    def close(self) -> None:
        connection, self.connection = self.connection, None
        self.session_id = None
        self.contract = None
        self.codec = None
        self._aac_previous = None
        if connection is not None:
            connection.close()


def build_model(config):
    return StarVlaPolicyModel(config)
