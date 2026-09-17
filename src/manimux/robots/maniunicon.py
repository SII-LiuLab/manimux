from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from manimux.clock import Clock
from manimux.types import RobotCommand, RobotState


class _ManiState(Protocol):
    joint_positions: np.ndarray
    gripper_state: np.ndarray


class _ArmInterface(Protocol):
    def connect(self) -> bool: ...

    def disconnect(self) -> bool: ...

    def get_state(self) -> _ManiState: ...

    def send_action(self, action: object) -> bool: ...

    def reset_to_init(self) -> bool: ...

    def stop(self) -> bool: ...


ActionFactory = Callable[..., object]
InterfaceFactory = Callable[[dict[str, object]], _ArmInterface]


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_maniunicon_arm_config(root: Path, path: Path) -> dict[str, object]:
    config_path = path if path.is_absolute() else root / path
    raw = _load_yaml(config_path)
    config = dict(raw["robot_interface"]["config"])

    urdf_path = config.get("urdf_path")
    if isinstance(urdf_path, str) and not Path(urdf_path).is_absolute():
        config["urdf_path"] = str(root / urdf_path)
    package_dirs = config.get("urdf_package_dirs")
    if isinstance(package_dirs, list):
        config["urdf_package_dirs"] = [
            str(root / item) if isinstance(item, str) and not Path(item).is_absolute() else item
            for item in package_dirs
        ]
    return config


class ManiUniConMeshcatDualArmDriver:
    """Dual-arm ManiMux driver composed from two ManiUniCon Meshcat interfaces.

    The dependency is injected for tests and imported lazily by ``from_config_file``.
    Each arm keeps its own Meshcat scene; ManiMux provides the synchronized command
    boundary and one combined state snapshot.
    """

    REQUIRED_GROUPS = {"left_arm", "right_arm", "left_gripper", "right_gripper"}

    def __init__(
        self,
        *,
        left: _ArmInterface,
        right: _ArmInterface,
        action_factory: ActionFactory,
        group_dims: dict[str, int],
        clock: Clock,
    ) -> None:
        if set(group_dims) != self.REQUIRED_GROUPS:
            raise ValueError(f"ManiUniCon dual-arm groups must be {sorted(self.REQUIRED_GROUPS)}")
        if group_dims["left_gripper"] != 1 or group_dims["right_gripper"] != 1:
            raise ValueError("ManiUniCon gripper groups must each have dimension 1")
        self._left = left
        self._right = right
        self._action_factory = action_factory
        self._group_dims = dict(group_dims)
        self._clock = clock
        self._connected = False
        self._sequence = 0

    @classmethod
    def from_config_file(
        cls,
        config_path: Path,
        group_dims: dict[str, int],
        clock: Clock,
    ) -> ManiUniConMeshcatDualArmDriver:
        raw = _load_yaml(config_path)
        root = Path(raw["maniunicon_root"]).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"ManiUniCon root does not exist: {root}")

        arm_paths: dict[str, Path] = {}
        for side in ("left", "right"):
            arm_paths[side] = Path(raw[side])

        # An editable/no-deps ManiUniCon checkout is sufficient for this optional adapter.
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        meshcat_module: Any = importlib.import_module("maniunicon.robot_interface.meshcat")
        storage_module: Any = importlib.import_module(
            "maniunicon.utils.shared_memory.shared_storage"
        )
        interface_factory: InterfaceFactory = meshcat_module.MeshcatInterface
        return cls(
            left=interface_factory(_resolve_maniunicon_arm_config(root, arm_paths["left"])),
            right=interface_factory(_resolve_maniunicon_arm_config(root, arm_paths["right"])),
            action_factory=storage_module.RobotAction,
            group_dims=group_dims,
            clock=clock,
        )

    def connect(self) -> None:
        if not self._left.connect():
            raise RuntimeError("failed to connect left ManiUniCon Meshcat interface")
        if not self._right.connect():
            self._left.disconnect()
            raise RuntimeError("failed to connect right ManiUniCon Meshcat interface")
        self._connected = True

    def get_state(self) -> RobotState:
        if not self._connected:
            raise RuntimeError("ManiUniCon dual-arm simulator is not connected")
        left = self._left.get_state()
        right = self._right.get_state()
        groups = {
            "left_arm": np.asarray(left.joint_positions, dtype=np.float64),
            "right_arm": np.asarray(right.joint_positions, dtype=np.float64),
            "left_gripper": np.asarray(left.gripper_state, dtype=np.float64).reshape(-1),
            "right_gripper": np.asarray(right.gripper_state, dtype=np.float64).reshape(-1),
        }
        for name, dim in self._group_dims.items():
            if groups[name].shape != (dim,):
                raise RuntimeError(
                    f"ManiUniCon state group {name!r} has shape {groups[name].shape}, "
                    f"expected {(dim,)}"
                )
        self._sequence += 1
        return RobotState(
            groups=groups,
            monotonic_ns=self._clock.now_ns(),
            sequence=self._sequence,
        )

    def _arm_action(self, command: RobotCommand, side: str) -> object:
        timestamp = time.time()
        return self._action_factory(
            control_mode="joint",
            joint_positions=command.groups[f"{side}_arm"].copy(),
            gripper_state=command.groups[f"{side}_gripper"].copy(),
            timestamp=timestamp,
            target_timestamp=timestamp,
        )

    def _hold_interface(self, interface: _ArmInterface) -> bool:
        state = interface.get_state()
        timestamp = time.time()
        action = self._action_factory(
            control_mode="joint",
            joint_positions=np.asarray(state.joint_positions, dtype=np.float64).copy(),
            gripper_state=np.asarray(state.gripper_state, dtype=np.float64).reshape(-1).copy(),
            timestamp=timestamp,
            target_timestamp=timestamp,
        )
        return interface.send_action(action)

    def send_command(self, command: RobotCommand) -> None:
        if not self._connected:
            raise RuntimeError("ManiUniCon dual-arm simulator is not connected")
        # Both actions are constructed and validated before either interface is touched.
        left_action = self._arm_action(command, "left")
        right_action = self._arm_action(command, "right")
        if not self._left.send_action(left_action):
            raise RuntimeError("left ManiUniCon interface rejected command")
        if not self._right.send_action(right_action):
            self._hold_interface(self._left)
            raise RuntimeError("right ManiUniCon interface rejected command; both arms stopped")

    def home(self) -> None:
        left_ok = self._left.reset_to_init()
        right_ok = self._right.reset_to_init()
        if not left_ok or not right_ok:
            raise RuntimeError("failed to reset both ManiUniCon arms")

    def stop(self) -> None:
        if self._connected:
            # Meshcat's upstream stop() uses velocity limits whose YAML schema is
            # inconsistent across robots. Position hold is the simulator-safe stop.
            self._hold_interface(self._left)
            self._hold_interface(self._right)

    def close(self) -> None:
        if self._connected:
            self._left.disconnect()
            self._right.disconnect()
            self._connected = False
