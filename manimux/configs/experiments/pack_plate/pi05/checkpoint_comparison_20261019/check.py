"""Offline catalog, artifact and step-execution checks; never connect hardware."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.embodiments.sensor import build_sensor
from manimux.experiment_list import load_experiment_list
from manimux.kinematics.poses import matrix_pose
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.policies.xpolicylab.codec import GroupLayout, decode_policy_actions
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.executors.smooth import SmoothExecutor
from manimux.runtime.inference import build_inference_strategy
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.safety import SafetyGuard
from manimux.types import ActionContext, ActionHorizon, ObservationSnapshot, RobotState
from XPolicyLab.policy.Pi_05.pack_plate_contract import validate_artifacts

HERE = Path(__file__).resolve().parent


def check_position_limits(cfg, robot):
    """Align command boxes and independent measured boxes across all layers."""
    safety = cfg["executor"]["command_safety"]
    assert safety["check_measured_position_limits"] is True
    for group, model in robot.kinematics.models.items():
        arm = model.arm
        assert arm.limit_margin_deg == 3.0
        lower, upper = arm.joint_position_limits()
        expected = (
            lower + np.deg2rad(arm.limit_margin_deg),
            upper - np.deg2rad(arm.limit_margin_deg),
        )
        expected_measured = (lower + np.deg2rad(0.5), upper - np.deg2rad(0.5))
        component = robot.arm_components[group]
        assert component.controller.check_measured_position_limits is True
        settings = component.controller.settings[component.channel]
        hardware = settings.joint_limits
        for index, key in enumerate(("position_lower", "position_upper")):
            np.testing.assert_allclose(hardware[index], expected[index], atol=1e-14, rtol=0)
            np.testing.assert_array_equal(hardware[index], safety[key][group][:7])
            np.testing.assert_allclose(
                settings.measured_joint_limits[index], expected_measured[index], atol=1e-14, rtol=0,
            )
            np.testing.assert_array_equal(
                settings.measured_joint_limits[index], safety["measured_" + key][group][:7],
            )
        # Null QP margins inherit the arm margin in both execution modes.
        tuning = (cfg["execution"]["ik"]["diff_ik"]
                  if cfg["execution"]["ik"]["mode"] == "step"
                  else cfg["policy"]["adapter"].get("diff_ik"))
        if tuning is not None:
            assert tuning["limit_margin_deg"] in (None, 3.0)
            assert tuning["j67_margin_deg"] in (None, 3.0)


def check_rates(cfg, robot, adapter):
    """Exercise executor saturation and the runtime guard without dispatch."""
    motion = cfg["executor"]["motion_limits"]
    assert motion["arm"]["max_velocity"] == np.pi
    safety = cfg["executor"]["command_safety"]
    for group in cfg["robot"]["group_dims"]:
        np.testing.assert_array_equal(safety["max_velocity"][group][:7], [np.pi] * 7)
    for solver in getattr(adapter, "diff_solvers", {}).values():
        assert solver.config.max_velocity_rad_s == np.pi / 2
        assert solver.vmax == 90.0
    home = yaml.safe_load(Path(cfg["robot"]["config"]).read_text())["home"]["joints_deg"]
    groups = {g: np.r_[np.deg2rad(q), 0.5] for g, q in home.items()}
    state = RobotState(groups, 0, 0)
    dt_s = 1 / cfg["robot"]["control_hz"]
    executor = (SmoothExecutor(cfg["executor"]["smooth"], dt_s)
                if cfg["executor"]["type"] == "smooth" else DirectExecutor(motion, dt_s))
    executor.reset(state)
    targets = {g: np.tile(q, (2, 1)) for g, q in groups.items()}
    for q in targets.values():
        q[:, 0] += 0.5
    reference = ActionHorizon(0, round(dt_s * 1e9), "offline-rate", targets)
    command = executor.step(0, state, reference)
    for g, q in command.groups.items():
        np.testing.assert_allclose(q[0] - groups[g][0], np.pi * dt_s, atol=1e-14)
    guard = SafetyGuard(cfg["robot"]["group_dims"], None, **safety, control_dt_s=dt_s)
    guard.reset(state)
    guard.validate_command(command)


def check_joint(cfg, adapter):
    """Exercise native wire targets and RTC aperture inversion without FK/IK."""
    assert cfg["policy"]["options"]["action_format"] == "joint"
    assert cfg["policy"]["action_decoding"] == "inline"
    assert cfg["inference"]["expected_decode_s"] == 0
    rows = {
        "left_arm": np.tile(np.r_[np.linspace(-0.7, 0.7, 7), 0.6], (32, 1)),
        "right_arm": np.tile(np.r_[np.linspace(0.2, -0.4, 7), 0.8], (32, 1)),
    }
    layouts = tuple(GroupLayout(f"{side}_arm", side, 7, 1) for side in ("left", "right"))
    wire = [
        {key: value for side in ("left", "right") for key, value in (
            (f"{side}_arm_joint_state", rows[f"{side}_arm"][i, :7]),
            (f"{side}_ee_joint_state", rows[f"{side}_arm"][i, 7:]),
        )} for i in range(32)
    ]
    raw = decode_policy_actions(wire, layouts=layouts, format="joint")
    state = RobotState({g: r[0] for g, r in rows.items()}, 0, 0)
    chunk = adapter.decode_action(raw, ActionContext(1, 0, 0, observation_state=state))
    assert chunk.action_space == "joint_position" and chunk.runtime_trajectory is None
    for group in rows:
        np.testing.assert_array_equal(chunk.groups[group][:, :7], rows[group][:, :7])
    request = RtcInferenceRequest(
        session_id="offline-joint", request_seq=1, observation_time_ns=0,
        deadline_ns=1_000_000_000, observation=ObservationSnapshot(state),
        action_condition=np.concatenate(list(chunk.groups.values()), axis=1),
        condition_weights=np.ones(32), action_condition_space="joint_position",
    )
    restored = adapter.prepare_request(request).action_condition
    np.testing.assert_allclose(restored, np.concatenate(list(rows.values()), axis=1))
    for sensor_cfg in cfg["sensors"]:
        # Constructor-only verification includes the optional direct main camera.
        sensor = build_sensor(sensor_cfg, SystemClock())
        if sensor_cfg.get("component") == "main_camera":
            assert sensor._camera is None
            assert Path(sensor_cfg["options"]["device_path"]).is_absolute()
        sensor.close()


def check_step(cfg, original, robot, adapter):
    """Verify retained contracts and exercise pose decode -> step QP -> direct command."""
    for key in ("robot", "sensors", "camera_server", "policy_server"):
        assert cfg[key] == original[key], key
    for key in ("motion_limits", "command_safety"):
        assert cfg["executor"][key] == original["executor"][key], key
    for key in ("task", "max_control_steps", "warmup_before_start"):
        assert cfg["run"][key] == original["run"][key], key
    expected_inference = deepcopy(original["inference"])
    expected_inference.update(expected_decode_s=0.0, decode_forecast_size=0)
    assert cfg["inference"] == expected_inference
    assert (cfg["policy"]["adapter"]["gripper_mapping"]
            == original["policy"]["adapter"]["gripper_mapping"])
    assert cfg["execution"]["ik"]["diff_ik"] == original["policy"]["adapter"]["diff_ik"]
    assert cfg["executor"]["type"] == "direct"
    assert cfg["policy"]["action_decoding"] == "inline"
    assert cfg["executor"]["motion_limits"]["arm"]["max_velocity"] == np.pi
    assert cfg["executor"]["motion_limits"]["arm"]["max_acceleration"] is None

    home = yaml.safe_load(Path(cfg["robot"]["config"]).read_text())["home"]["joints_deg"]
    groups = {g: np.r_[np.deg2rad(q), 0.5] for g, q in home.items()}
    state = RobotState(groups, 0, 0)
    row = {}
    for name, model in robot.kinematics.models.items():
        side = name.removesuffix("_arm")
        pose = matrix_pose(model.fk(groups[name]))
        pose[0] += 0.001
        row[f"{side}_ee_pose"] = pose
        row[f"{side}_ee_joint_state"] = np.array([0.6])
    adapter.decode_pose_only = True
    chunk = adapter.decode_action(
        [row] * cfg["policy"]["horizon_policy_steps"],
        ActionContext(1, 0, 0, ik_seed=state, observation_state=state),
    )
    assert chunk.action_space == POSE_SPACE
    assert chunk.runtime_trajectory is None
    dt_ns = round(1e9 / cfg["robot"]["control_hz"])
    pipe = ExecutionPipeline(
        DirectExecutor(cfg["executor"]["motion_limits"], dt_ns / 1e9),
        kinematics=robot.kinematics, ik_options=cfg["execution"]["ik"],
        control_dt_ns=dt_ns, motion_limits=cfg["executor"]["motion_limits"],
    )
    pipe.warmup(groups)
    for solver in pipe.resolver.diff.solvers.values():
        assert solver.config.max_velocity_rad_s == np.pi / 2
        assert solver.vmax == 90.0
    reference = ActionHorizon(0, dt_ns, "offline-step", {
        g: rows[:1] for g, rows in chunk.groups.items()
    }, action_space=POSE_SPACE)
    output = pipe.step(0, state, reference, last_command=groups)
    assert output.ik_status["success"], output.ik_status
    for group, q in output.command.groups.items():
        assert q.shape == (8,) and np.isfinite(q).all()
        assert np.max(np.abs(q[:7] - groups[group][:7])) <= np.pi / 2 * dt_ns / 1e9 + 1e-10

    if cfg["inference"]["algorithm"] == "rtc":
        order = cfg["policy"]["adapter"]["group_order"]
        condition = np.concatenate([chunk.groups[g] for g in order], axis=1)
        weights = np.ones(chunk.horizon_steps)
        weights[-1] = 0
        request = RtcInferenceRequest(
            session_id="offline-check", request_seq=1, observation_time_ns=0,
            deadline_ns=1_000_000_000, observation=ObservationSnapshot(state),
            action_condition=condition, condition_weights=weights,
            action_condition_space=POSE_SPACE,
        )
        prepared = adapter.prepare_request(request)
        assert prepared.action_condition.shape == (32, 16)
        np.testing.assert_allclose(prepared.action_condition[:-1, [7, 15]], 0.6)
        np.testing.assert_array_equal(prepared.action_condition[-1], 0)
    return output.ik_status["elapsed_ms"]


def main():
    catalog = load_experiment_list(HERE / "catalog.yaml")
    assert len(catalog.choices) == 32 and len(catalog.policy_services) == 16
    timings = []
    for choice in catalog.choices.values():
        cfg = choice.config
        # Match serve's offline assembly construction; never call connect().
        robot = build_robot(cfg["robot"], SystemClock())
        check_position_limits(cfg, robot)
        build_inference_strategy(cfg)
        adapter = build_policy_adapter(
            cfg["robot"], cfg["policy"], kinematics=robot.kinematics,
            motion_limits=cfg["executor"]["motion_limits"],
        )
        adapter.validate(cfg["robot"], cfg["policy"])
        check_rates(cfg, robot, adapter)
        validate_artifacts(cfg["policy_server"])
        if cfg["policy_server"]["action_type"] == "joint":
            check_joint(cfg, adapter)
        if cfg["execution"]["ik"]["mode"] == "step":
            original = catalog.choices[choice.id.removesuffix("_step_ik")].config
            timings.append(check_step(cfg, original, robot, adapter))
        print(choice.id)
    print(f"Validated {len(catalog.choices)} choices / {len(catalog.policy_services)} policies.")
    print("Step decode/QP/direct and RTC pose codec passed; "
          f"max sampled solve {max(timings):.2f} ms.")
    print("Native joint wire decode, RTC aperture roundtrip and inert main-camera construction passed.")
    print("All choices align command/IK margins at 3 degrees and measured margins at 0.5 degrees.")
    print("Executor/runtime arm limits are pi rad/s; chunk/step QP limits are pi/2 rad/s.")
    print(f"Default: {catalog.default_id}; no services or hardware started.")
    print("CPU samples do not establish sustained control timing or physical task success.")


if __name__ == "__main__":
    main()
