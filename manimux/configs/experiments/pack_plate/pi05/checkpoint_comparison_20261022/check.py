"""Offline catalog, artifact and step-execution checks; never connect hardware."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.embodiments.sensor import build_sensor
from manimux.experiment_list import load_experiment_list
from manimux.kinematics.poses import matrix_pose
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.handoff import WaypointHandoff
from manimux.policy_adapter.pose_output import POSE_SPACE
from manimux.policies.xpolicylab.codec import GroupLayout, decode_policy_actions
from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.executors.smooth import SmoothExecutor
from manimux.runtime.inference import build_inference_strategy
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.runtime.safety import SafetyGuard
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionContext, ActionHorizon, ObservationSnapshot, RobotState
from XPolicyLab.policy.Pi_05.pack_plate_contract import validate_artifacts

HERE = Path(__file__).resolve().parent


def check_handoff_comparison(catalog):
    """Only handoff settings and recording/display identity may differ."""
    for umi in (0, 25, 50, 75, 90, 95, 100):
        stem = f"ratio_umi{umi:03}_tele{100-umi:03}_29999_rtc16"
        base = catalog.choices[stem].config
        assert base["inference"]["handoff"] == "blend"
        assert base["inference"]["blend_policy_steps"] == 0
        assert base["inference"]["handoff_skip_steps"] == 0
        assert base["execution"]["ik"]["mode"] == "chunk"
        for mode in ("blend2", "waypoint"):
            variant = deepcopy(catalog.choices[f"{stem}_{mode}"].config)
            expected = deepcopy(base["inference"])
            expected.update(handoff="waypoint" if mode == "waypoint" else "blend",
                            blend_policy_steps=0 if mode == "waypoint" else 2)
            assert variant["inference"] == expected
            for key in ("condition", "output_dir"):
                assert variant["run"][key] != base["run"][key]
                variant["run"][key] = base["run"][key]
            variant["robogui"]["policy_label"] = base["robogui"]["policy_label"]
            variant["inference"] = base["inference"]
            assert variant == base, stem


def check_waypoint(cfg, robot, adapter):
    """Decode a synthetic handoff with real DiffIK and commit without dispatch."""
    assert adapter.supports_waypoint_handoff
    limits = adapter.waypoint_handoff.config
    assert limits.max_pos_speed_m_s == 0.4 and limits.max_rot_speed_rad_s == 1.2
    home = yaml.safe_load(Path(cfg["robot"]["config"]).read_text())["home"]["joints_deg"]
    groups = {g: np.r_[np.deg2rad(q), 0.5] for g, q in home.items()}
    state = RobotState(groups, 1_000_000_000, 1)
    row = {}
    for name, model in robot.kinematics.models.items():
        side = name.removesuffix("_arm")
        row[f"{side}_ee_pose"] = matrix_pose(model.fk(groups[name]))
        row[f"{side}_ee_joint_state"] = np.array([0.5])
    raw = [row] * cfg["policy"]["horizon_policy_steps"]
    first = adapter.decode_action(raw, ActionContext(
        1, state.monotonic_ns, state.monotonic_ns, observation_state=state,
        ik_seed=state, ik_seed_source="observation_state",
    ))
    timeline = ActionTimeline(cfg["robot"]["group_dims"])
    settings = dict(max_plan_age_ns=2_000_000_000, current_command=groups, blend_steps=0)
    assert timeline.commit(first, now_ns=state.monotonic_ns, **settings).accepted
    observation_ns = state.monotonic_ns + 100_000_000
    handoff_ns = observation_ns + 90_000_000
    context = ActionContext(
        2, observation_ns, observation_ns, observation_state=state,
        execution_time_ns=handoff_ns,
        handoff_reference=timeline.handoff_reference(handoff_ns, first.dt_ns),
    )
    seed_ns = adapter.decode_seed_time_ns(context)
    context = replace(context, ik_seed=RobotState(
        WaypointHandoff.sample_reference(context.handoff_reference, seed_ns), seed_ns, 2,
    ), ik_seed_source="active_reference")
    target = deepcopy(row)
    for side in ("left", "right"):
        target[f"{side}_ee_pose"][0] += 0.05
    chunk = adapter.decode_action([target] * len(raw), context)
    assert chunk.metadata["handoff_limited_rows"] > 0
    for name in groups:
        np.testing.assert_array_equal(chunk.runtime_trajectory.groups[name][0],
                                      context.ik_seed.groups[name])
    assert timeline.commit(chunk, now_ns=chunk.handoff.time_ns - 1, **settings).accepted


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
    expected_acceleration = 160.0 if cfg["execution"]["ik"]["mode"] == "step" else 80.0
    assert motion["arm"]["max_acceleration"] == expected_acceleration
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
        np.testing.assert_allclose(q[0] - groups[g][0], expected_acceleration * dt_s**2, atol=1e-14)
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
    assert cfg["executor"]["command_safety"] == original["executor"]["command_safety"]
    expected_motion = deepcopy(original["executor"]["motion_limits"])
    expected_motion["arm"]["max_acceleration"] = 160.0
    assert cfg["executor"]["motion_limits"] == expected_motion
    for key in ("task", "max_control_steps", "warmup_before_start"):
        assert cfg["run"][key] == original["run"][key], key
    expected_inference = deepcopy(original["inference"])
    expected_inference.update(expected_decode_s=0.0, decode_forecast_size=0)
    assert cfg["inference"] == expected_inference
    assert (cfg["policy"]["adapter"]["gripper_mapping"]
            == original["policy"]["adapter"]["gripper_mapping"])
    expected_diff = {**original["policy"]["adapter"]["diff_ik"], "max_acceleration_rad_s2": 80.0, "elbow_guard_deg": 1.0}
    assert cfg["execution"]["ik"]["diff_ik"] == expected_diff
    assert cfg["executor"]["type"] == "direct"
    assert cfg["policy"]["action_decoding"] == "inline"
    assert cfg["executor"]["motion_limits"]["arm"]["max_velocity"] == np.pi
    assert cfg["executor"]["motion_limits"]["arm"]["max_acceleration"] == 160.0

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
    assert len(catalog.choices) == 60 and len(catalog.policy_services) == 14
    check_handoff_comparison(catalog)
    timings = []
    for choice in catalog.choices.values():
        cfg = choice.config
        # Match serve's offline assembly construction; never call connect().
        robot = build_robot(cfg["robot"], SystemClock())
        assert robot.controller.tracking_error_policy == "report"
        assert robot.controller._max_tracking == np.deg2rad(10.0)
        check_position_limits(cfg, robot)
        build_inference_strategy(cfg)
        adapter = build_policy_adapter(
            cfg["robot"], cfg["policy"], kinematics=robot.kinematics,
            motion_limits=cfg["executor"]["motion_limits"],
        )
        adapter.validate(cfg["robot"], cfg["policy"])
        if cfg["inference"]["handoff"] == "waypoint":
            check_waypoint(cfg, robot, adapter)
        check_rates(cfg, robot, adapter)
        validate_artifacts(cfg["policy_server"])
        if cfg["policy_server"]["action_type"] == "joint":
            check_joint(cfg, adapter)
        if cfg["execution"]["ik"]["mode"] == "step":
            if choice.id.endswith("_rtc12_beta10_skip1_blend2_step_ik"):
                baseline_id = choice.id.split("_rtc12_", 1)[0] + "_rtc16"
                original = deepcopy(catalog.choices[baseline_id].config)
                original["inference"].update(
                    chunk_policy_steps=12, handoff_skip_steps=1, blend_policy_steps=2,
                )
                original["inference"]["rtc"].update(min_execute_policy_steps=12, beta=10.0)
            elif choice.id == "ratio_umi050_tele050_29999_rtc12_beta10_skip1_step_ik":
                original = deepcopy(catalog.choices["ratio_umi050_tele050_29999_rtc16"].config)
                original["inference"].update(chunk_policy_steps=12, handoff_skip_steps=1)
                original["inference"]["rtc"].update(min_execute_policy_steps=12, beta=10.0)
            elif choice.id in {
                "ratio_umi090_tele010_29999_rtc12_beta10_skip1_blend1_step_ik",
                "ratio_umi095_tele005_29999_rtc12_beta10_skip1_blend1_step_ik",
                "ratio_umi100_tele000_29999_rtc12_beta10_skip1_blend1_step_ik",
            }:
                baseline_id = choice.id.split("_rtc12_", 1)[0] + "_rtc16"
                original = deepcopy(catalog.choices[baseline_id].config)
                original["inference"].update(
                    chunk_policy_steps=12, handoff_skip_steps=1, blend_policy_steps=1,
                )
                original["inference"]["rtc"].update(min_execute_policy_steps=12, beta=10.0)
            else:
                original = catalog.choices[choice.id.removesuffix("_step_ik")].config
            timings.append(check_step(cfg, original, robot, adapter))
        print(choice.id)
    print(f"Validated {len(catalog.choices)} choices / {len(catalog.policy_services)} policies.")
    print("RTC 16 None / Blend 2 / Waypoint match across seven checkpoints; "
          "speed-limited waypoint decode and continuous timeline commit passed.")
    print("Step decode/QP/direct and RTC pose codec passed; "
          f"max sampled solve {max(timings):.2f} ms.")
    print("All choices align command/IK margins at 3 degrees and measured margins at 0.5 degrees.")
    print("Executor/runtime arm limits are pi rad/s; chunk/step QP limits are pi/2 rad/s.")
    print("Step QP acceleration: 80 rad/s^2; outer executor: 160. Other executors: 80.")
    print(f"Default: {catalog.default_id}; no services or hardware started.")
    print("CPU samples do not establish sustained control timing or physical task success.")


if __name__ == "__main__":
    main()
