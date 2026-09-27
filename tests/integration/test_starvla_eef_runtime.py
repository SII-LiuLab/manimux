"""Real EEF serving/IK/runtime boundaries with explicitly synthetic model outputs.

The StarVLA Model, shared WebSocket server, worker, YAM geometry, decoder,
scheduler, executors and recorder are real. Only PolicyServerWrapper.predict_action
is replaced. These tests are neither checkpoint inference nor robot task success.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml
import zarr

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policies.decoder import ActionDecoderClient
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.pi05.yam_eef import Pi05YamEefAdapter
from manimux.runtime import build_runtime
from manimux.types import ActionContext, InferenceResponse, RobotState

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "manimux/configs/experiments/offline/starvla/yam_eef_contract.yaml"


@pytest.fixture
def setup(tmp_path):
    run = tmp_path / "synthetic-model-no-weights"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/fixture.pt"
    checkpoint.touch()
    (run / "config.yaml").write_text(
        "framework: {name: QwenOFT}\ndatasets: {vla_data: {include_state: true}}\n"
    )
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump({"paths": {"checkpoint": str(checkpoint)}}))
    config = load_config(EXPERIMENT, local=station)
    config["run"]["max_control_steps"] = 300
    robot = build_robot(config["robot"], SystemClock())
    assert not robot._connected
    return config, robot


def target_poses(config, robot):
    """Generate reachable, nonconstant reference poses with the real YAM FK."""
    targets = {}
    for group, initial in config["robot"]["options"]["initial_groups"].items():
        rows = []
        for step in range(50):
            q = np.asarray(initial).copy()
            q[0] += 0.015 * np.sin(step * 0.13)
            q[4] += 0.008 * np.sin(step * 0.17)
            q[-1] += 0.03 * np.sin(step * 0.11)
            rows.append(np.r_[matrix_pose(robot.kinematics.models[group].fk(q)), q[-1]])
        targets[group] = np.asarray(rows)
    return targets


def state_of(config):
    return RobotState(
        {g: np.asarray(q).copy() for g, q in config["robot"]["options"]["initial_groups"].items()},
        time.monotonic_ns(),
        0,
    )


@contextmanager
def served_model(config, robot, monkeypatch):
    """Replace expensive inference only, and label the fixture in HELLO metadata."""
    monkeypatch.syspath_prepend(str(ROOT / "XPolicyLab"))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    from XPolicyLab.policy.starVLA.model import Model

    targets = target_poses(config, robot)
    semantics = config["policy"]["adapter"]["action_semantics"]
    group_order = tuple(config["robot"]["group_dims"])

    class SyntheticRuntime:
        def __init__(self, **kwargs):
            self.calls = []
            self.metadata = dict(
                framework="QwenOFT",
                action_chunk_size=50,
                ckpt_path=str(config["policy_server"]["checkpoint_path"]),
                available_unnorm_keys=["synthetic_fixture"],
                default_unnorm_key="synthetic_fixture",
                runtime_contract=dict(
                    version=1,
                    image_color_order="rgb",
                    state_input="raw_env",
                    state_normalization="training_transform",
                    action_output="unnormalized_env",
                    action_dim=16,
                    state_dim=16,
                ),
            )

        def predict_action(self, *, examples, **kwargs):
            self.calls.append(deepcopy(examples))
            time.sleep(0.04)  # Exercise observation-to-response delay and latency trimming.
            output = []
            for example in examples:
                assert len(example["image"]) == 3
                state = np.asarray(example["state"])
                assert state.shape == (1, 16) and np.isfinite(state).all()
                arms = []
                for i, group in enumerate(group_order):
                    anchor = pose_matrix(state[0, i * 8 : i * 8 + 7])
                    rows = []
                    for row in targets[group]:
                        target = pose_matrix(row[:7])
                        if semantics == "delta_observation_tool_xyz_wxyz":
                            target = np.linalg.inv(anchor) @ target
                        elif semantics == "delta_observation_base_xyz_wxyz":
                            target[:3, 3] -= anchor[:3, 3]
                            target[:3, :3] = target[:3, :3] @ anchor[:3, :3].T
                        rows.append(np.r_[matrix_pose(target), row[-1]])
                    arms.append(rows)
                output.append(np.concatenate(arms, axis=1))
            return {"actions": np.asarray(output, dtype=np.float32)}

    module = ModuleType("deployment.model_server.policy_wrapper")
    module.PolicyServerWrapper = SyntheticRuntime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    model = Model(config["policy_server"])
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host="127.0.0.1",
            port=0,
            model_metadata={"validation_fixture": "synthetic_eef_no_weights"},
        ),
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        asyncio.run_coroutine_threadsafe(server.start(), loop).result(timeout=10)
        port = server._server.sockets[0].getsockname()[1]
        config["policy"]["options"]["server"] = f"ws://127.0.0.1:{port}"
        config["policy"]["expected_backend"]["model"]["validation_fixture"] = (
            "synthetic_eef_no_weights"
        )
        yield model, targets
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=10)
        asyncio.run_coroutine_threadsafe(loop.shutdown_default_executor(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
        assert not thread.is_alive()


@pytest.mark.parametrize(
    "mode,executor,semantics,schedule",
    [
        ("inline", "smooth", "absolute_per_arm_base_xyz_wxyz", "async"),
        ("process", "smooth", "absolute_per_arm_base_xyz_wxyz", "async"),
        ("parallel", "smooth", "absolute_per_arm_base_xyz_wxyz", "async"),
        ("parallel", "mpc", "absolute_per_arm_base_xyz_wxyz", "async"),
        ("parallel", "smooth", "delta_observation_base_xyz_wxyz", "async"),
        ("parallel", "smooth", "delta_observation_tool_xyz_wxyz", "async"),
        ("parallel", "smooth", "absolute_per_arm_base_xyz_wxyz", "serial"),
        ("inline", "smooth", "absolute_per_arm_base_xyz_wxyz", "ensemble"),
    ],
)
def test_eef_shared_server_runtime(
    setup, tmp_path, monkeypatch, mode, executor, semantics, schedule
):
    config, robot = setup
    config["policy"]["action_decoding"] = "inline" if mode == "inline" else "process"
    config["policy"]["adapter"].update(parallel_ik=mode == "parallel", action_semantics=semantics)
    config["policy"]["expected_backend"]["model"]["action_semantics"] = semantics
    config["policy_server"]["eef"]["semantics"] = semantics
    config["policy"]["expected_backend"]["model"]["eef"]["semantics"] = semantics
    if mode == "inline":
        config["inference"]["expected_decode_s"] = 0.0
    config["executor"]["type"] = executor
    if schedule == "serial":
        config["inference"].update(
            inference_schedule="serial", action_start_mode="first_step_when_ready"
        )
    elif schedule == "ensemble":
        config["inference"].update(algorithm="act_temporal_ensemble", blend_policy_steps=0)
    with served_model(config, robot, monkeypatch) as (model, targets):
        runtime = build_runtime(config, tmp_path / "runtime")
        result = runtime.run()
        assert result.success and result.steps == 300
        assert result.accepted_plans >= 3
        assert model.policy.calls and model.include_state
        assert runtime._worker._process.exitcode == 0
        assert not runtime._robot._connected
        if runtime._decoder:
            assert all(not p.is_alive() for p in runtime._decoder._processes)
        events = [
            json.loads(line)
            for line in (result.episode_dir / "events.jsonl").read_text().splitlines()
        ]
        accepted = [e for e in events if e["kind"] == "plan_accepted"]
        assert len(accepted) == result.accepted_plans
        rejected = [e for e in events if e["kind"] in {"plan_rejected", "inference_rejected"}]
        # Configured offline IK warmup runs before the control clock starts.
        assert not rejected and result.rejected_plans == 0
        data = zarr.open_group(str(result.episode_dir / "data.zarr"), mode="r")
        ticks = data["ticks/monotonic_ns"][:]
        assert len(ticks) == 300 and np.all(np.diff(ticks) > 0)
        for _, plan in data["plans"].groups():
            raw = plan["canonical_raw"]
            assert raw.attrs["metadata"]["decoded_policy_steps"] == 20
            assert raw.attrs["dt_ns"] == 33_333_333
            for group in targets:
                rows = raw[group][:]
                assert rows.shape == (20, 7)
                for q, expected in zip(rows, targets[group][:20], strict=True):
                    np.testing.assert_allclose(
                        robot.kinematics.models[group].fk(q), pose_matrix(expected[:7]), atol=3e-4
                    )
                    assert abs(q[-1] - expected[-1]) < 1e-6
            if mode == "parallel":
                assert set(raw.attrs["metadata"]["decode_partition_ms"]) == set(targets)
        for group in targets:
            for stage in ("state", "scheduled", "optimized", "command"):
                assert np.isfinite(data[f"ticks/{stage}/{group}"][:]).all()
            assert np.ptp(data[f"ticks/command/{group}"][:, 0]) > 1e-4
        if mode != "inline":
            decoded = [e for e in events if e["kind"] == "decode_submitted"]
            assert decoded
            if semantics.startswith("delta_observation_"):
                assert all(e["seed_source"] == "observation_state" for e in decoded)
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                dict(
                    model_execution="synthetic fixture; no checkpoint weights",
                    geometry="YAM",
                    hardware_used=False,
                    mode=mode,
                    schedule=schedule,
                    executor=executor,
                    semantics=semantics,
                    accepted=result.accepted_plans,
                    rejected=result.rejected_plans,
                    steps=result.steps,
                    raw_horizon=50,
                    decoded_prefix=20,
                ),
                indent=2,
            )
            + "\n"
        )


def test_parallel_decoder_matches_pi_prefix_and_rejects_whole_chunk(setup):
    config, robot = setup
    adapter = build_policy_adapter(config["robot"], config["policy"], kinematics=robot.kinematics)
    state = state_of(config)
    raw = {"format": "pose", "actions": target_poses(config, robot)}
    context = ActionContext(1, state.monotonic_ns, state.monotonic_ns, measured_state=state)
    expected = adapter.decode_action(raw, context)
    pi = Pi05YamEefAdapter(config["robot"], config["policy"], kinematics=robot.kinematics)
    pi_chunk = pi.decode_action(raw, context)
    for group in expected.groups:
        np.testing.assert_allclose(expected.groups[group], pi_chunk.groups[group], atol=1e-8)
    decoder = ActionDecoderClient(config["robot"], config["policy"], adapter)
    try:
        decoder.start()
        for seq, invalid in enumerate((False, True), start=1):
            now = time.monotonic_ns()
            if invalid:
                raw["actions"]["right_arm"][3, 0] = 100.0  # Unreachable TCP target.
            response = InferenceResponse("test", seq, now, 0.0, raw, observation_time_ns=now)
            context = ActionContext(seq, now, now, measured_state=state)
            deadline = now + 10_000_000_000
            decoder.submit(response, context, deadline)
            result = None
            while result is None and time.monotonic_ns() < deadline:
                result = decoder.poll()
                time.sleep(0.002)
            assert result is not None
            if invalid:
                assert (
                    result.chunk is None
                    and "Pose IK failed: group=right_arm step=3" in result.error
                )
            else:
                assert result.error is None
                for group in expected.groups:
                    np.testing.assert_allclose(
                        result.chunk.groups[group], expected.groups[group], atol=1e-8
                    )
    finally:
        decoder.close()
    assert all(not process.is_alive() for process in decoder._processes)
