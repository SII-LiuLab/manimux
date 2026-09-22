"""New registry -> robot -> UMI adapter, with native IK and fake hardware."""

import importlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.end_effector import GripperState
from manimux.embodiments.robot import RobotModel, build_robot, robot_parameters
from manimux.embodiments.sensor import sensor_parameters
from manimux.kinematics.tianji import TianjiKinematics
from manimux.policies import ActionDecoderClient
from manimux.policies.base import action_interval
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.umi_dp.history import WindowSnapshot
from manimux.policy_adapter.umi_dp.tianji import matrix_pose
from manimux.runtime.edge import EdgeRuntime
from manimux.types import (
    ActionContext,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
    RobotCommand,
    RobotState,
    SensorFrame,
)

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY = ROOT / "manimux/configs/embodiment/robot/tianji_taccap.yaml"
DIFF_EXPERIMENT = (
    ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp_diff.yaml"
)
Q = np.radians([21.8, -41, -4.74, -63.67, 10.15, 14.72, 7.68])


def configured():
    cfg = load_config(ROOT / "manimux/configs/experiments/pass_ball/tianji_umi_dp_default.yaml")
    cfg["robot"] = robot_parameters(
        type="tianji_taccap", config=ASSEMBLY, group_dims={"left_arm": 8, "right_arm": 8}
    )
    # A test identity satisfies the existing handshake contract; no model is loaded.
    cfg["policy"]["adapter"]["deployment_bound"] = True
    for key in ("kinematics", "kinematics_options", "right_kinematics_options"):
        cfg["policy"]["options"].pop(key, None)
    cfg["policy"]["expected_backend"]["model"].update(
        checkpoint_sha256="offline-test",
        training_config_sha256="offline-test",
        checkpoint_path="offline-test",
        weight_key="ema",
        rgb_normalize=True,
        action_horizon=cfg["policy"]["horizon_steps"],
        action_dt_s=action_interval(cfg["policy"]),
        first_action_offset_s=cfg["policy"]["adapter"]["first_action_offset_s"],
        observation_period_s=cfg["policy"]["adapter"]["observation_period_s"],
    )
    return cfg


def actions(model, horizon):
    steps = []
    for i in range(horizon):
        target = Q.copy()
        target[0] += np.radians(0.02 * (i + 1))
        poses = model.fk({name: np.r_[target, 0.8] for name in model.models})
        steps.append(
            {f"{side}_ee_pose": matrix_pose(poses[f"{side}_arm"]) for side in ("left", "right")}
            | {f"{side}_ee_joint_state": [0.8] for side in ("left", "right")}
        )
    return steps


def context(now):
    state = RobotState({name: np.r_[Q, 0.8] for name in ("left_arm", "right_arm")}, now, 1)
    return ActionContext(1, now, now, execution_time_ns=now, measured_state=state)


def test_runtime_shares_robot_kinematics_without_opening_devices(tmp_path):
    cfg = configured()
    runtime = EdgeRuntime(cfg, tmp_path)
    assert runtime._adapter.robot_kinematics is runtime._robot.kinematics
    assert runtime._robot.model.kinematics is runtime._robot.kinematics
    assert runtime._robot.controller._robot is None
    assert all(sensor._camera is None for sensor in runtime._robot.sensors.values())
    runtime._robot.close()


def test_assembled_fk_ik_matches_original_policy_solver():
    model = RobotModel.from_config(ASSEMBLY)
    for side in ("left", "right"):
        legacy = TianjiKinematics(
            arm=side,
            end_effector="umi_follower",
            joint_limits_deg={6: [-58, 58]} if side == "right" else None,
        )
        assembled = model.groups[f"{side}_arm"].kinematics
        for delta in (0.0, 0.0003, -0.0003):
            target_q = Q.copy()
            target_q[0] += delta
            pose = legacy.fk(target_q, 0.8)
            np.testing.assert_allclose(assembled.fk(np.r_[target_q, 0.8]), pose, atol=1e-12)
            old_ok, old_q = legacy.ik(pose, Q, 0.8)
            new = assembled.ik(pose, np.r_[Q, 0.8], fixed_coordinates={"gripper": 0.8})
            assert new.converged == old_ok
            assert old_ok
            np.testing.assert_allclose(new.joints[:7], old_q, atol=1e-10)
        # Preserve branch rejection, not just easy FK/IK round trips.
        far_q = Q.copy()
        far_q[0] += np.radians(10)
        pose = legacy.fk(far_q, 0.8)
        old_ok, _ = legacy.ik(pose, Q, 0.8)
        new = assembled.ik(pose, np.r_[Q, 0.8], fixed_coordinates={"gripper": 0.8})
        assert not new.converged and not old_ok
        assert new.reason == "branch_jump"


def test_observation_and_action_stay_in_each_arm_base():
    cfg = configured()
    robot = build_robot(cfg["robot"], SystemClock())
    adapter = build_policy_adapter(cfg["robot"], cfg["policy"], kinematics=robot.kinematics)
    now = time.monotonic_ns()
    state = context(now).measured_state
    previous = ObservationSnapshot(RobotState(state.groups, now - 100_000_000, 0), {})
    frames = {
        name: SensorFrame(name, np.zeros((8, 8, 3), dtype=np.uint8), now, 1)
        for name in adapter.cameras.values()
    }
    request = InferenceRequest("test", 1, now, now + 10**9, WindowSnapshot(state, frames, previous))
    prepared = adapter.prepare_request(request)
    for side in ("left", "right"):
        expected = robot.fk({f"{side}_arm": np.r_[Q, 0.8]})[f"{side}_arm"]
        np.testing.assert_allclose(
            prepared.xpolicylab_state[f"{side}_ee_pose"], matrix_pose(expected)
        )
    result = adapter.decode_action(
        actions(robot.kinematics, cfg["policy"]["horizon_steps"]), context(now)
    )
    assert result.action_space == "joint_position"
    for values in result.groups.values():
        assert values.shape == (cfg["policy"]["horizon_steps"], 8)
        np.testing.assert_allclose(values[:, -1], 0.8)


@pytest.mark.parametrize("backend", ["analytic", "diff"])
def test_spawned_action_decode_matches_inline(backend):
    cfg = configured()
    cfg["policy"]["adapter"]["ik_backend"] = backend
    if backend == "diff":
        from manimux.policy_adapter.umi_dp.ik_config import bind_diff_ik_profile

        bind_diff_ik_profile(cfg)
    adapter = build_policy_adapter(cfg["robot"], cfg["policy"])
    steps = actions(adapter.robot_kinematics, cfg["policy"]["horizon_steps"])
    decoder = ActionDecoderClient(cfg["robot"], cfg["policy"], adapter)
    try:
        decoder.start()
        now = time.monotonic_ns()
        ctx = context(now)
        expected = adapter.decode_action(steps, ctx)
        response = InferenceResponse("test", 1, now, 1.0, steps, observation_time_ns=now)
        decoder.submit(response, ctx, now + 30_000_000_000)
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            result = decoder.poll()
            time.sleep(0.01)
        assert result is not None and result.error is None
        for name in expected.groups:
            np.testing.assert_allclose(result.chunk.groups[name], expected.groups[name], atol=1e-10)
    finally:
        decoder.close()


@pytest.mark.parametrize("execute", [False, True])
def test_decoded_actions_reach_shared_controller_only_when_enabled(monkeypatch, execute):
    from test_tianji_robot import SDK

    sdk = SDK()
    subscribe = sdk.subscribe

    def feedback(buffer):
        data = subscribe(buffer)
        for frame in data["outputs"]:
            frame["fb_joint_pos"] = np.degrees(Q).tolist()
        return data

    sdk.subscribe = feedback
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *args, **kwargs: (
            SimpleNamespace(Marvin_Robot=lambda: sdk, DCSS=object)
            if name == "manimux.embodiments.arm.tianji.sdk.marvin.fx_robot"
            else real_import(name, *args, **kwargs)
        ),
    )
    cfg = configured()
    cfg["robot"]["options"] = {"hardware": {"ip": "192.0.2.1"}, "execute": execute}
    clock = SystemClock()
    robot = build_robot(cfg["robot"], clock)
    for tool in robot.end_effectors.values():
        monkeypatch.setattr(tool, "connect", lambda: None)
        monkeypatch.setattr(tool, "get_state", lambda: GripperState(0.8, clock.now_ns() / 1e9))
    adapter = build_policy_adapter(cfg["robot"], cfg["policy"], kinematics=robot.kinematics)
    try:
        robot.connect()
        chunk = adapter.decode_action(
            actions(robot.kinematics, cfg["policy"]["horizon_steps"]), context(clock.now_ns())
        )
        robot.send_command(
            RobotCommand({n: q[0] for n, q in chunk.groups.items()}, clock.now_ns(), None)
        )
        if execute:
            assert [entry[1] for entry in sdk.batches[-1]] == ["A", "B"]
            for side, target in zip(("left_arm", "right_arm"), sdk.batches[-1], strict=True):
                np.testing.assert_allclose(target[2], np.degrees(chunk.groups[side][0, :7]))
        else:
            assert sdk.batches == [] and sdk.modes == [0, 0]
    finally:
        robot.close()


def test_new_recipe_keeps_original_timing_and_control_envelopes():
    old = load_config(ROOT / "manimux/configs/experiments/pass_ball/tianji_umi_dp_default.yaml")
    new = load_config(ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml")
    assert new["robot"]["type"] == "tianji_taccap" and new["robot"]["config"] == ASSEMBLY
    assert new["robot"]["group_dims"] == old["robot"]["group_dims"]
    assert new["robot"]["control_hz"] == old["robot"]["control_hz"]
    assert action_interval(new["policy"]) == action_interval(old["policy"])
    assert (
        new["policy"]["adapter"]["first_action_offset_s"]
        == old["policy"]["adapter"]["first_action_offset_s"]
    )
    forecast_keys = {"expected_decode_s", "decode_forecast_size", "decode_forecast_mode"}
    assert {key: value for key, value in new["inference"].items() if key not in forecast_keys} == {
        key: value for key, value in old["inference"].items() if key not in forecast_keys
    }
    assert new["inference"]["expected_decode_s"] == 0.06
    assert new["inference"]["decode_forecast_size"] == 10
    assert new["inference"]["decode_forecast_mode"] == "max"
    assert new["executor"] == old["executor"]
    assert not new["robot"]["options"]["execute"]
    assert not new["robot"]["options"]["end_effector_control"]
    robot = build_robot(new["robot"], SystemClock())
    assert robot.controller._robot is None
    robot.close()


def test_diff_ik_experiment_is_complete_and_matches_motion_profile():
    from manimux.policy_adapter.umi_dp.history import HistoryStrategy
    from manimux.policy_adapter.umi_dp.ik_config import bind_diff_ik_profile

    config = load_config(DIFF_EXPERIMENT)
    options = config["policy"]["adapter"]
    motion = config["executor"]["motion_limits"]["arm"]
    assert options["ik_backend"] == "diff"
    assert options["diff_ik"]["max_velocity_rad_s"] == motion["max_velocity"]
    assert options["diff_ik"]["dt_max_s"] == motion["max_step_dt_s"]
    assert options["diff_ik"]["check_j67"]
    assert config["policy"]["action_decoding"] == "process"
    bind_diff_ik_profile(config)
    HistoryStrategy(config)


def test_renaming_camera_frames_preserves_pixels_time_and_sequence(monkeypatch):
    from manimux.embodiments.sensor.camera_server.timestamped import TimestampedCameraSensor

    now = 1_000_000_000
    monkeypatch.setattr(time, "time_ns", lambda: now)
    clock = SimpleNamespace(now_ns=lambda: now)
    sensor = TimestampedCameraSensor(
        sensor_parameters(
            name="camera_server",
            driver="unused",
            options={
                "camera_names": ["left_wrist"],
                "output_names": {"left_wrist": "left_wrist_camera"},
            },
        ),
        clock,
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    bundle = {"frames": {"left_wrist": image}, "timestamps": {"left_wrist": now / 1e9}}
    sensor.client = SimpleNamespace(try_recv_bundle=lambda: bundle)
    sensor.offset_ns = 0
    renamed = sensor.read()["left_wrist_camera"]
    assert renamed.name == "left_wrist_camera"
    assert renamed.data is image
    assert renamed.capture_monotonic_ns == now and renamed.sequence == now
    assert sensor.read()["left_wrist_camera"] is renamed


def test_serve_entry_does_not_import_legacy_recovery(tmp_path):
    from manimux.session import RuntimeSessionService

    cfg = configured()
    cfg["viewer"]["enabled"] = True
    service = RuntimeSessionService(cfg, tmp_path)
    assert service._ready_metadata()["recovery"] == {"available": False}


def test_checkpoint_binding_preserves_assembly_path_when_relocated(tmp_path, monkeypatch):
    import runpy
    import sys

    report = dict(configured()["policy"]["expected_backend"]["model"])
    module = "XPolicyLab.policy.UMI_DP.artifact_identity"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(validate_deployment=lambda _: report))
    # The launcher adjusts sys.path; keep the surrounding test process unchanged.
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = tmp_path / "experiment" / "run.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "umi_dp_tianji_server.py",
            "--experiment",
            str(ROOT / "manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml"),
            "--local",
            str(ROOT / "manimux/configs/local/tianji_taccap.example.yaml"),
            "--bind-runtime-config",
            str(output),
        ],
    )
    runpy.run_path(str(ROOT / "manimux/servers/umi_dp.py"), run_name="__main__")
    bound = load_config(output)
    assert bound["robot"]["type"] == "tianji_taccap" and bound["robot"]["config"] == ASSEMBLY
    assert bound["policy"]["adapter"]["deployment_bound"]
    assert output.with_name("run-server.yaml").is_file()
