"""CAD consistency and real offline SDK assembly checks; no hardware access."""

import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from manimux.embodiments.arm.tianji import TianjiSDKKinematics
from manimux.embodiments.end_effector.taccap import TacCapGeometry
from manimux.embodiments.robot import RobotModel
from manimux.kinematics.end_effector import Frame

CONFIG = Path(__file__).resolve().parents[2] / "manimux/configs/embodiment/robot/tianji_taccap.yaml"


def test_geometry_and_mount_match_existing_cad_data():
    path = (
        Path(__file__).resolve().parents[2]
        / "manimux/embodiments/end_effector/taccap/assets/umi_follower/end_effector.yaml"
    )
    spec = yaml.safe_load(path.read_text())

    def transform(section):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("xyz", section["rpy"]).as_matrix()
        result[:3, 3] = section["xyz"]
        return result

    tool = TacCapGeometry()
    np.testing.assert_allclose(tool.tcp_transform(np.array([0.0])), transform(spec["tcp"]))
    assembly = yaml.safe_load(CONFIG.read_text())
    mount = Frame.model_validate(assembly["components"]["left_end_effector"]["mount"]).matrix()
    np.testing.assert_allclose(mount, transform(spec["mount"]), atol=1e-15)
    for aperture in (0.0, 0.5, 1.0):
        combined = mount @ tool.tcp_transform(np.array([aperture]))
        np.testing.assert_allclose(combined[:3, 3], [-0.036745, 0, 0.16945], atol=1e-15)
        np.testing.assert_allclose(combined[:3, :3], [[0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=1e-15)


native = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"},
    reason="bundled libKine targets Linux x86-64",
)


@native
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("custom", [False, True])
def test_native_assembly_roundtrip(side, custom, tmp_path):
    spec = yaml.safe_load(CONFIG.read_text())
    for component in spec["components"].values():
        component["config"] = str((CONFIG.parent / component["config"]).resolve())
    entry = spec["components"][f"{side}_end_effector"]
    if custom:
        entry["mount"]["xyz"] = (np.array(entry["mount"]["xyz"]) + [0.01, -0.02, 0.03]).tolist()
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
        tcp[:3, 3] = [0.12, -0.01, 0.02]
        entry["options"] = {"tcp_transform": tcp.tolist()}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(spec))
    group = RobotModel.from_config(path).groups[f"{side}_arm"]
    model = group.kinematics
    mount = group.mount.matrix()
    tool = group.end_effector.geometry
    sign = 1 if side == "left" else -1
    joints = np.radians([sign * 21.8, -41, sign * -4.74, -63.67, sign * 10.15, 14.72, sign * 7.68])
    q = np.r_[joints, 0.7]
    expected = TianjiSDKKinematics(side).fk_flange(joints) @ mount @ tool.tcp_transform(q[-1:])
    target = model.fk(q)
    np.testing.assert_allclose(target, expected, atol=1e-12)
    seed = q.copy()
    seed[0] += np.radians(0.1)
    seed[-1] = 0.2
    result = model.ik(target, seed, fixed_coordinates={"gripper": 0.7})
    assert result.converged, result.reason
    assert result.joints.shape == (8,) and result.joints[-1] == 0.7
    np.testing.assert_allclose(model.fk(result.joints), target, atol=1e-7)
    assert model.base_frame == f"tianji_{side}_base"
    assert [c.name for c in model.coordinates] == [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
    mount[:] = 0
    np.testing.assert_allclose(model.fk(q), target, atol=1e-12)
    with pytest.raises(NotImplementedError):
        model.ik(target, seed, fixed_coordinates={})
    unreachable = target.copy()
    unreachable[0, 3] = 10
    rejected = model.ik(unreachable, seed, fixed_coordinates={"gripper": 0.7})
    assert not rejected.converged and rejected.joints is None


@native
def test_grouped_native_tcp_roundtrip():
    grouped = RobotModel.from_config(CONFIG).kinematics
    q = {}
    for side, sign in (("left", 1), ("right", -1)):
        q[f"{side}_arm"] = np.r_[
            np.radians([sign * 21.8, -41, sign * -4.74, -63.67, sign * 10.15, 14.72, sign * 7.68]),
            0.6,
        ]
    poses = grouped.fk(q)
    results = grouped.ik(poses, q, fixed_coordinates={n: {"gripper": 0.6} for n in q})
    for name, result in results.items():
        assert result.converged, result.reason
        np.testing.assert_allclose(grouped.models[name].fk(result.joints), poses[name], atol=1e-7)
