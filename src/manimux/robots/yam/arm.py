import threading

import numpy as np

try:
    from i2rt.robots.utils import GripperType
except ImportError as exc:
    raise ImportError(
        "YAM hardware control requires i2rt commit 5d47b358; "
        "install it with the command documented in README.md."
    ) from exc

from manimux.robots.yam.base import Robot


class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    def __init__(self, channel="can0", **hardware_options):
        from i2rt.robots.get_robot import get_yam_robot

        if "gripper_force_limit" in hardware_options:
            import inspect

            force = hardware_options["gripper_force_limit"]
            if not np.isfinite(force) or force <= 0:
                raise ValueError("gripper_force_limit must be finite and positive")
            if "gripper_force_limit" not in inspect.signature(get_yam_robot).parameters:
                if force != 50.0:
                    raise ValueError("installed ManiMux i2rt fixes gripper force at 50 N")
                hardware_options.pop("gripper_force_limit")
        if "arm_type" in hardware_options:
            from i2rt.robots.utils import ArmType

            hardware_options["arm_type"] = ArmType.from_string_name(hardware_options["arm_type"])
        gripper_type = GripperType.LINEAR_4310
        if "gripper_type" in hardware_options:
            gripper_type = GripperType.from_string_name(hardware_options.pop("gripper_type"))
        self.robot = get_yam_robot(
            channel=channel, gripper_type=gripper_type, **hardware_options
        )

        # YAM has 7 joints (6 arm joints + 1 gripper)
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = self.get_joint_state()  # robot stays where it was when reboot
        # self._joint_state = np.zeros(7)  # robot goes immediately to reset position (avoid using)
        self._joint_velocities = np.zeros(7)  # 7 joints
        self._gripper_state = 0.0  # didn't use because joint_state includes gripper position

    def num_dofs(self) -> int:
        return 7  # YAM has 7 DOFs

    def get_joint_state(self) -> np.ndarray:
        # Get actual joint positions from I2RT robot (7 joints total)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we have exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")

        self._joint_state = joint_pos
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs(), (
            f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"
        )

        dt = 0.01
        self._joint_velocities = (joint_state - self._joint_state) / dt
        self._joint_state = joint_state

        # Command the I2RT robot with all 7 joints (6 arm + 1 gripper)
        self.command_joint_pos(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }

    def get_joint_pos(self):
        # Get 7 joints from I2RT robot (6 arm + 1 gripper)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we return exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")
        return joint_pos

    def command_joint_pos(self, target_pos):
        # Ensure we send exactly 7 joints to the I2RT robot
        if len(target_pos) > 7:
            target_pos = target_pos[:7]
        elif len(target_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            target_pos = np.pad(target_pos, (0, 7 - len(target_pos)), "constant")
        self.robot.command_joint_pos(np.array(target_pos))

    def close(self) -> None:
        """Join both i2rt threads before closing their CAN file descriptor."""

        native = self.robot
        stop_event = getattr(native, "_stop_event", None)
        server_thread = getattr(native, "_server_thread", None)
        motor_chain = getattr(native, "motor_chain", None)
        if stop_event is None or server_thread is None or motor_chain is None:
            raise RuntimeError("unsupported i2rt backend: safe close hooks are unavailable")

        stop_event.set()
        server_thread.join(timeout=2.0)
        if server_thread.is_alive():
            raise RuntimeError("i2rt robot server thread did not stop; CAN left open")

        motor_chain.running = False
        control_threads = []
        for thread in threading.enumerate():
            target = getattr(thread, "_target", None)
            if getattr(target, "__self__", None) is motor_chain:
                control_threads.append(thread)
        for thread in control_threads:
            thread.join(timeout=2.0)
        if any(thread.is_alive() for thread in control_threads):
            raise RuntimeError("i2rt motor control thread did not stop; CAN left open")

        motor_chain.close()
        print("Robot closed with all torques set to zero.")


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
