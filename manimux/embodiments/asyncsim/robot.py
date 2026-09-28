"""Simulator-backed grouped state and single-target robot commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from manimux.types import RobotCommand, RobotState


class AsyncSimRobot:
    """Structural robot interface; RobotBase itself owns physical arm assemblies."""

    def __init__(
        self, client: Any, group_dims: Mapping[str, int], state_keys: Mapping[str, str],
        *, state_stream: str = "proprio.joint_state", env_index: int = 0,
        command_ttl_s: float = 0.2,
    ) -> None:
        self.client = client
        self.group_dims = dict(group_dims)
        self.state_keys = dict(state_keys)
        if not self.group_dims or set(self.group_dims) != set(self.state_keys):
            raise ValueError("state keys must exactly match robot groups")
        if any(dim <= 0 for dim in self.group_dims.values()) or command_ttl_s <= 0:
            raise ValueError("group dimensions and command TTL must be positive")
        self.state_stream = state_stream
        self.env_index = env_index
        self.command_ttl_s = command_ttl_s
        self._sequence = 0
        self._command_seq = 0
        self._last_snapshot: dict[str, Any] | None = None

    def connect(self) -> None:
        self.client.connect()
        declared = self.client.health().get("canonical_group_dims")
        if declared != self.group_dims:
            raise ValueError("AsyncSim canonical group layout does not match ManiMux configuration")

    def reset(self) -> None:
        self._sequence = 0
        self._command_seq = 0
        self._last_snapshot = None

    def use_snapshot(self, snapshot: dict[str, Any]) -> RobotState:
        entry = snapshot["packets"][self.state_stream]
        packet = entry["packet"]
        if packet is None:
            raise RuntimeError("AsyncSim state stream is missing")
        if packet["episode_id"] != self.client.episode_id:
            raise RuntimeError("state packet belongs to another episode")
        payload = packet["payload"]
        source = payload.get(self.env_index, payload.get(str(self.env_index)))
        if not isinstance(source, Mapping):
            raise ValueError("state payload is missing the configured environment")
        groups = {}
        for name, dim in self.group_dims.items():
            values = np.asarray(source[self.state_keys[name]], dtype=np.float64).reshape(-1)
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError(f"invalid AsyncSim state group {name!r}")
            groups[name] = values
        anchor = snapshot["clock_anchor"]
        capture_ns = packet.get("wall_ts_ns") or anchor["monotonic_ns"] + round(
            (packet["capture_ts"] - anchor["sim_ts"]) * 1_000_000_000
        )
        self._sequence = packet["seq"]
        self._last_snapshot = snapshot
        return RobotState(groups, capture_ns, self._sequence)

    def get_state(self) -> RobotState:
        return self.use_snapshot(self.client.read_snapshot([self.state_stream]))

    def send_command(self, command: RobotCommand) -> dict[str, Any]:
        snapshot = self._last_snapshot
        if snapshot is None or self.client.episode_id is None:
            raise RuntimeError("read robot state before sending a command")
        if set(command.groups) != set(self.group_dims) or any(
            command.groups[name].shape != (dim,) for name, dim in self.group_dims.items()
        ):
            raise ValueError("command layout does not match AsyncSim robot")
        now = self.client.health()["episode"]["sim_ts"]
        self._command_seq += 1
        issued = {
            "episode_id": self.client.episode_id,
            "command_seq": self._command_seq,
            "issued_sim_ts": now,
            "apply_after_sim_ts": now,
            "expires_at_sim_ts": now + self.command_ttl_s,
            "plan_id": command.plan_id or "hold",
            "action_space": "joint_position",
            "groups": {name: values.tolist() for name, values in command.groups.items()},
        }
        ack = self.client.submit_command(issued)
        if not ack["accepted"]:
            raise RuntimeError(f"AsyncSim command rejected: {ack['reason']}")
        return {**ack, **{key: issued[key] for key in (
            "issued_sim_ts", "apply_after_sim_ts", "expires_at_sim_ts",
        )}}

    def home(self) -> None:
        raise NotImplementedError("AsyncSim homing requires a configured simulator reset")

    def stop(self) -> None:
        self.client.close()

    def close(self) -> None:
        self.client.close()
