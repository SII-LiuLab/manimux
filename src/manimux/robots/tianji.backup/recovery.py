"""Exclusive, idle-session ownership of Tianji drag and homing."""

from __future__ import annotations

import threading
from typing import Any

from manimux.clock import SystemClock
from manimux.config import ManiMuxConfig

from .driver import build_robot
from .teleop_drag import DEFAULT_TELEOP_ROOT, TeleopDragProcess


class TianjiRecovery:
    def __init__(self, config: ManiMuxConfig) -> None:
        self._config = config
        self.available = (
            config.robot.driver == "tianji_dual"
            and config.viewer.robot_adapter == "tianji"
            and config.robot.options.get("execute") is True
        )
        self._drag = TeleopDragProcess(
            config.viewer.tianji_teleop_root or DEFAULT_TELEOP_ROOT,
            str(config.robot.options.get("robot_ip", "192.168.1.190")),
        )
        self._home_thread: threading.Thread | None = None
        self._home_error = ""
        self.error = ""
        self.ack = ""

    @property
    def busy(self) -> bool:
        return self._drag.busy or self._home_thread is not None

    def metadata(self) -> dict[str, object]:
        return {
            "available": self.available,
            "state": "homing" if self._home_thread is not None else self._drag.state,
            "busy": self.busy,
            "arm": self._drag.arm,
            "error": self.error or self._drag.error,
            "ack": self.ack,
        }

    def update(self, control: dict[str, Any]) -> None:
        self._drag.poll()
        if self._home_thread is not None and not self._home_thread.is_alive():
            self._home_thread.join()
            self._home_thread = None
            self.error = self._home_error
        # The viewer must continuously renew drag ownership; a lost connection
        # stops drag. Homing remains the driver's bounded home operation.
        if self._drag.busy and not control.get("recovery_lease", False):
            self._drag.stop()
        request_id = str(control.get("recovery_request_id", ""))
        if not request_id or request_id == self.ack:
            return
        self.ack = request_id
        request = str(control.get("recovery_request", ""))
        try:
            if not self.available:
                raise ValueError("Recovery requires an executing Tianji runtime service")
            if request == "stop":
                self._drag.stop()
                return
            if self.busy:
                raise ValueError("Exit drag / wait for homing before another recovery action")
            self.error = ""
            if request == "home":
                self._drag.error = ""
                self._drag.state = "idle"
                self._home_error = ""
                self._home_thread = threading.Thread(target=self._home, daemon=True)
                self._home_thread.start()
            elif request.startswith("drag:"):
                arm = request.removeprefix("drag:")
                active = self._config.robot.options.get("active_arms", ["left_arm", "right_arm"])
                if not isinstance(active, list | tuple):
                    raise ValueError("active_arms must list left_arm and/or right_arm")
                allowed = "".join(
                    a for a, g in (("A", "left_arm"), ("B", "right_arm")) if g in active
                )
                if arm not in {"A", "B", "AB"} or any(a not in allowed for a in arm):
                    raise ValueError("Requested drag arms are not enabled in active_arms")
                if not control.get("recovery_lease", False):
                    raise ValueError("Drag requires an active viewer connection")
                self._drag.start(arm)
            else:
                raise ValueError(f"Unknown recovery action: {request}")
        except (OSError, ValueError, RuntimeError) as exc:
            self.error = str(exc)

    def _home(self) -> None:
        robot = None
        errors: list[str] = []
        try:
            robot = build_robot(self._config.robot, SystemClock())
            robot.connect(recover_errors=True)
            robot.home()
        except Exception as exc:
            errors.append(str(exc))
        finally:
            if robot is not None:
                for cleanup in (robot.stop, robot.close):
                    try:
                        cleanup()
                    except Exception as exc:
                        errors.append(str(exc))
            self._home_error = "; ".join(errors)

    def close(self) -> None:
        self._drag.close()
        if self._home_thread is not None:
            self._home_thread.join()
