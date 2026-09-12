"""Build collection policies over the shared ManiMux execution backend."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import CollectionBackend, FollowerView, load_backend_config
from .camera.interface import CameraDriver, CameraMode
from .camera.mock_camera import MockCamera
from .config import (
    StationConfig,
    controller_channel_for,
    gello_variant,
    is_passive_gello,
    leader_joint_signs_for,
)
from .robot.mock_robot import MockTeleop


@dataclass
class ArmUnit:
    name: str
    robot: FollowerView
    agent: object


def build_arm_units(cfg: StationConfig, mock=False, followers_only=False) -> list[ArmUnit]:
    if followers_only:
        raise RuntimeError("Use manimux serve for policy deployment")
    mock = mock or cfg.robot.type == "mock"
    config = load_backend_config(cfg, mock=mock)
    if abs(config.policy.action_dt_s * cfg.control_hz - 1.0) > 1e-6:
        raise ValueError("station control_hz must match runtime policy.action_dt_s")
    backend = CollectionBackend(
        config,
        mock=mock,
        config_path=cfg.manimux_config,
        execution_mode=cfg.execution_mode,
    )
    units = []
    leaders = []
    try:
        backend.connect()
        for robot in cfg.robot.robots:
            side = robot.type.removeprefix("yam_")
            follower = FollowerView(backend, f"{side}_arm")
            if mock:
                policy = MockTeleop(follower)
            else:
                from .robot.yam_adapter import YamLeaderArm, YamLeaderPolicy

                controller = cfg.robot.controller_for(robot)
                channel = controller_channel_for(controller)
                passive = is_passive_gello(controller.type)
                if passive:
                    from .robot.passive_gello import PassiveGelloLeader

                    variant, variant_side = gello_variant(controller.type)
                    leader = PassiveGelloLeader(
                        channel=channel,
                        num_arm_joints=6,
                        gripper_config=tuple(cfg.robot.leader_gripper or [6, 0.7, 0.0]),
                        joint_signs=leader_joint_signs_for(
                            controller, cfg.robot.leader_joint_signs
                        ),
                        gello_type=variant,
                        side=variant_side,
                    )
                else:
                    leader = YamLeaderArm(
                        channel=channel,
                        arm_type=cfg.robot.arm_type,
                        gripper_type=cfg.robot.leader_gripper_type,
                        ee_mass=cfg.robot.ee_mass,
                        bilateral_kp=cfg.robot.bilateral_kp,
                    )
                leaders.append(leader)
                policy = YamLeaderPolicy(
                    leader,
                    follower,
                    bilateral_kp=0.0 if passive else cfg.robot.bilateral_kp,
                    gripper_mode=cfg.robot.leader_gripper_mode,
                    control_hz=cfg.control_hz,
                    gripper_close_duration_s=cfg.robot.gripper_close_duration_s,
                )
            units.append(ArmUnit(side, follower, policy))
        return units
    except BaseException:
        for leader in reversed(leaders):
            close = getattr(leader, "close", None) or getattr(leader, "stop", None)
            if close:
                close()
        backend.close()
        raise


def build_cameras_from_config(cfg: StationConfig, mock: bool = False) -> list[CameraDriver]:
    if mock or cfg.robot.type == "mock":
        cams = cfg.cameras or [None]  # default to a single mock top camera if none configured
        out: list[CameraDriver] = []
        for c in cams:
            if c is None:
                out.append(MockCamera(name="top", role="top"))
            else:
                mode = CameraMode(c.mode)
                out.append(
                    MockCamera(name=c.name, role=c.role, mode=mode, width=c.width, height=c.height)
                )
        return out

    from .camera.registry import build_cameras

    return build_cameras(cfg.cameras)
