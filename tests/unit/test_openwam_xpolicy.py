"""OpenWAM XPolicy contracts; no checkpoint, GPU, CAN or camera required."""

import argparse
import asyncio
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "XPolicyLab"))

from manimux.config import load_config
from manimux.integrations.openwam_yam.policy_plugin import SEMANTICS, OpenWAMYamAdapter
from manimux.integrations.xpolicylab.policy_plugin import XPolicyLabWsPolicyModel
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)
from XPolicyLab.policy.OpenWAM.model import (
    Model,
    _configure_deploy_runtime,
    validate_deployment,
)
from XPolicyLab.policy.OpenWAM.training import build_command


class Kinematics:
    num_arm_joints = 6
    fail = False

    def fk(self, joints, grip):
        pose = np.eye(4)
        pose[:3, 3] = joints[:3]
        pose[:3, :3] = Rotation.from_rotvec(joints[3:]).as_matrix()
        return pose

    def ik(self, pose, seed, grip):
        return not self.fail, np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec()]


@pytest.fixture
def setup(monkeypatch):
    import manimux.kinematics

    kin = Kinematics()
    monkeypatch.setattr(manimux.kinematics, "build_kinematics", lambda *a, **k: kin)
    config = load_config(ROOT / "configs/openwam/yam/infra/manimux.yaml")
    config.robot.driver = "fake"
    adapter = OpenWAMYamAdapter(config.robot, config.policy)
    state = RobotState(
        {
            "left_arm": np.array([0.1, 0.2, 0.3, 0.2, -0.1, 0.4, 0.25]),
            "right_arm": np.array([-0.1, 0.2, 0.4, -0.3, 0.2, 0.1, 0.75]),
        },
        1,
        1,
    )
    frames = {
        name: SensorFrame(name, np.full((16, 16, 3), [200, 40, 10], np.uint8), 1, 1)
        for name in ("front_camera", "left_camera", "right_camera")
    }
    request = InferenceRequest(
        session_id="test",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=10**18,
        observation=ObservationSnapshot(state, frames),
        instruction="Assemble the screwdriver.",
    )
    model_config = dict(
        env_cfg_type="yam_dual",
        action_type="ee",
        observation_profile="yam_base",
        allow_dummy_policy=True,
        replan_steps=32,
    )
    return config, adapter, request, model_config, kin


def test_xpolicy_server_roundtrip(setup):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    config, adapter, request, model_config, _ = setup

    async def exercise():
        server = PolicyServer(Model(model_config), PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        config.policy.options["server"] = f"ws://127.0.0.1:{port}"

        def client():
            worker = XPolicyLabWsPolicyModel(config.policy)
            try:
                worker.reset("test")
                assert (
                    worker.capabilities().backend_metadata["model"]["observation_profile"]
                    == "yam_base"
                )
                raw = worker.infer(adapter.prepare_request(request))
                assert raw["action_semantics"] == SEMANTICS
                chunk = adapter.decode_action(raw, ActionContext(1, 1, 2))
                for group, values in request.observation.state.groups.items():
                    np.testing.assert_allclose(
                        chunk.groups[group], np.tile(values, (32, 1)), atol=1e-6
                    )
            finally:
                worker.close()

        try:
            await asyncio.to_thread(client)
        finally:
            await server.stop()

    asyncio.run(exercise())


def observation(adapter, request):
    from manimux.integrations.xpolicylab.obs_codec import build_layouts, encode_observation

    prepared = adapter.prepare_request(request)
    obs = encode_observation(
        request.observation,
        layouts=build_layouts(
            ["left_arm", "right_arm"],
            {"left_arm": "left", "right_arm": "right"},
            {"left_arm": 7, "right_arm": 7},
            gripper_dofs=1,
        ),
        camera_map=adapter.cameras,
        instruction=request.instruction,
        frequency=30,
    )
    obs["state"].update(prepared.xpolicylab_state)
    return obs


def test_batch_and_reset(setup):
    _, adapter, request, config, _ = setup
    model = Model(config)
    obs = observation(adapter, request)
    other = copy.deepcopy(obs)
    other["env_idx"] = 7
    other["state"]["left_ee_pose"][0] += 0.4
    model.update_obs_batch([obs, other])
    batch = model.get_action_batch([7, 0])
    assert batch[0][0]["left_ee_pose"][0] == pytest.approx(0.5)
    assert batch[1][0]["left_ee_pose"][0] == pytest.approx(0.1)
    model.reset()
    with pytest.raises(ValueError, match="No stored observation"):
        model.get_action()


def test_single_stream_uses_generate_not_generate_batch():
    class Engine:
        def __init__(self):
            self.calls = []

        def generate(self, conditions):
            self.calls.append(conditions)
            return {"actions": np.zeros((3, 20), dtype=np.float32)}

        def generate_batch(self, conditions):
            raise AssertionError("single-stream inference must not use generate_batch")

    class Policy:
        @staticmethod
        def _project_binary_dims(actions):
            return actions

    model = object.__new__(Model)
    model.eval_batch = False
    model.allow_dummy_policy = False
    model.replan_steps = 2
    model._engine = Engine()
    model._wam_policy = Policy()
    model._batch = {0: {"prompt": "put bottles"}}
    model._order = [0]
    model.observation_profile = "yam_base"
    model._conditions = lambda payload: payload
    model._eef20_chunk_to_native = lambda actions: actions.tolist()

    result = model.get_action()

    assert len(model._engine.calls) == 1
    assert len(result["actions"]) == 2
    assert result["action_semantics"] == SEMANTICS


def test_deploy_optimizations_are_single_stream_only():
    from omegaconf import OmegaConf

    single = OmegaConf.create(
        {
            "optimization": {
                "dit_cache": {"enabled": False},
                "compile": {"enabled": False},
                "decode_video": True,
            },
            "inference": {"inference_mode": "async"},
        }
    )
    assert not _configure_deploy_runtime(
        single,
        {"eval_batch": False, "compile_enabled": True, "dit_cache_enabled": True},
    )
    assert single.optimization.compile.enabled is True
    assert single.optimization.dit_cache.enabled is True
    assert single.optimization.decode_video is False
    assert single.inference.inference_mode == "sync"

    batch = copy.deepcopy(single)
    assert _configure_deploy_runtime(
        batch,
        {"eval_batch": True, "compile_enabled": True, "dit_cache_enabled": True},
    )
    assert batch.optimization.compile.enabled is False
    assert batch.optimization.dit_cache.enabled is False


def test_model_clips_continuous_gripper_predictions(setup):
    _, adapter, request, config, _ = setup
    model = Model(config)
    payload = model._encode_obs(observation(adapter, request))
    chunk = np.tile(np.asarray(payload["state"], dtype=np.float64), (2, 1))
    chunk[:, 9] = [-0.2, 1.2]
    chunk[:, 19] = [1.3, -0.3]

    actions = model._eef20_chunk_to_native(chunk)

    np.testing.assert_allclose(
        [step["left_ee_joint_state"][0] for step in actions],
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        [step["right_ee_joint_state"][0] for step in actions],
        [1.0, 0.0],
    )


@pytest.mark.parametrize("failure", ["semantics", "quaternion", "gripper", "horizon", "ik"])
def test_bad_actions_rejected(setup, failure):
    _, adapter, request, config, kin = setup
    model = Model(config)
    model.update_obs(observation(adapter, request))
    raw = model.get_action()
    if failure == "semantics":
        raw["action_semantics"] = "world"
    if failure == "quaternion":
        raw["actions"][0]["left_ee_pose"][3:] = 0
    if failure == "gripper":
        raw["actions"][0]["left_ee_joint_state"][0] = 2
    if failure == "horizon":
        raw["actions"].pop()
    if failure == "ik":
        kin.fail = True
    with pytest.raises(ValueError):
        adapter.decode_action(raw, ActionContext(1, 1, 2))


def test_profile_and_missing_checkpoint(setup):
    _, _, _, config, _ = setup
    with pytest.raises(ValueError, match="profile"):
        validate_deployment({**config, "observation_profile": "arx_x5_sim"})
    with pytest.raises(FileNotFoundError):
        validate_deployment({**config, "allow_dummy_policy": False})


def test_training_arguments(tmp_path):
    args = argparse.Namespace(
        bench="RoboDojo_real",
        robot="yam_dual",
        action="ee",
        run="unit",
        seed=0,
        gpu="0,1",
        dataset=tmp_path,
        output=tmp_path / "output",
        finetune=None,
        resume=None,
        overrides=["training.max_steps=1"],
    )
    command = build_command(args)
    assert "--nproc_per_node=2" in command
    assert "dataloader.variant=real" in command
    assert "dataloader.embodiment=yam_dual" in command
    assert "training.max_steps=1" in command
    args.overrides = ["dataloader.variant=sim"]
    with pytest.raises(ValueError, match="contract"):
        build_command(args)


def test_qz_launcher_disables_wandb_and_pins_training_budget():
    launcher = (ROOT / "scripts/training/train_openwam_yam_cluster.sh").read_text()
    profile = (ROOT / "scripts/training/train_openwam_yam_bottles_cluster.sh").read_text()
    setup = (ROOT / "scripts/training/setup_openwam_qz_env.sh").read_text()
    package = (ROOT / "XPolicyLab/policy/OpenWAM/OpenWAM/pyproject.toml").read_text()
    loader = (
        ROOT
        / "XPolicyLab/policy/OpenWAM/OpenWAM/openwam/model/video_backbone/wan/shared/core/loader/config.py"
    ).read_text()
    assert "WANDB_MODE=disabled" in launcher
    assert '"project.wandb.project=null"' in launcher
    assert "OPENWAM_MAX_STEPS:-30000" in profile
    assert "OPENWAM_SAVE_INTERVAL:-5000" in profile
    assert "OPENWAM_GPU_IDS:-0,1,2,3" in profile
    assert "OPENWAM_EXPECTED_EPISODES=50" in profile
    assert "OPENWAM_EXPECTED_FRAMES=35118" in profile
    assert "resolve_deploy_checkpoint_dir" in launcher
    assert '--checkpoint "${deploy_checkpoint_dir}" --check' in launcher
    assert "--system-site-packages" in setup
    assert "numpy==1.26.4" in setup
    assert "torch==" not in setup
    assert '"opencv-python-headless>=4.7,<5"' in package
    assert "from modelscope import snapshot_download" not in loader.split("class ModelConfig", 1)[0]
    assert "from huggingface_hub import snapshot_download" not in loader.split("class ModelConfig", 1)[0]


def test_checked_in_put_bottles_deployment_is_bound():
    config = load_config(
        ROOT / "configs/openwam/yam/infra/manimux-put-bottles-step30000.yaml"
    )
    identity = config.policy.expected_backend.model
    assert config.robot.control_hz == 100.0
    assert config.robot.options["start_joints"] == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
    assert config.policy.effective_action_dt_s == pytest.approx(1.0 / 30.0)
    assert config.policy.horizon_steps == 32
    assert config.policy.options["deployment_bound"] is True
    assert config.execution.inference_schedule == "serial"
    assert config.execution.chunk_steps == 12
    assert identity["checkpoint_file"] == "checkpoint_step_30000.safetensors"
    assert identity["action_horizon"] == 32
    assert config.execution.smooth.gripper.max_velocity == 1.0
    assert config.execution.smooth.gripper.max_acceleration == 12.0
