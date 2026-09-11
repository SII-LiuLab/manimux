"""Native recording -> HDF5 -> OpenWAM sample and deploy-stat validation."""

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_recording_to_training_sample(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "openwam_converter", ROOT / "scripts/datasets/prepare_openwam_yam_dataset.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    recording = tmp_path / "recording"
    recording.mkdir()
    count = 35
    (recording / "write_complete.flag").touch()
    (recording / "metadata.json").write_text(json.dumps({
        "num_frames": count, "control_hz": 30, "extra": {"eepose": {"enabled": True}}
    }))
    transforms = np.tile(np.eye(4), (count, 1, 1))
    transforms[:, 0, 3] = np.arange(count) * .001
    for side in ("left", "right"):
        np.save(recording / f"{side}-ee_transform.npy", transforms)
        np.save(recording / f"{side}-gripper_pos.npy", np.full((count, 1), .5))
    for filename in converter.CAMERAS.values():
        writer = cv2.VideoWriter(str(recording / filename), cv2.VideoWriter_fourcc(*"mp4v"), 30, (16, 16))
        assert writer.isOpened()
        for _ in range(count):
            writer.write(np.full((16, 16, 3), [10, 40, 200], np.uint8))
        writer.release()
    dataset_root = tmp_path / "dataset"
    target = dataset_root / "task/yam_dual/data/episode_000000.hdf5"
    assert converter.export_episode(recording, target, "move", 30) == count
    with pytest.raises(ValueError, match="overwrite"):
        converter.export_episode(recording, target, "move", 30)
    with pytest.raises(ValueError, match="frequency"):
        converter.export_episode(recording, dataset_root / "wrong.hdf5", "move", 20)

    from XPolicyLab.policy.OpenWAM.model import _resolve_openwam_root, validate_deployment
    _resolve_openwam_root(None)
    from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset, read_calibrated_eef20
    from XPolicyLab.policy.OpenWAM.process_data import prepare
    report = prepare(dataset_root, "yam_dual")
    assert report["sample_shapes"]["action"] == [32, 80]
    dataset = MultiTaskRoboDojoDataset(dataset_root, embodiment="yam_dual", variant="real",
                                     normalize_mode=None, unify_action_map=["0-9", "34-43"],
                                     color_jitter={"enabled": False})
    sample = dataset[0]
    # The first action is next achieved state, not the observation itself.
    assert float(sample["action"][0, 0]) == pytest.approx(.001)
    assert float(sample["proprio"][0, 0]) == pytest.approx(0)
    image = np.asarray(sample["video"][0])
    assert image[..., 0].mean() > image[..., 2].mean() + 100
    with h5py.File(target) as handle:
        eef = read_calibrated_eef20(handle, None, 0, 2, variant="real", embodiment="yam_dual")
        np.testing.assert_allclose(eef[:, 0], [0, .001])

    # Check the exported stats against the exact deployment validation path.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    import shutil
    shutil.copyfile(report["stats"], checkpoint / "normalization_stats.npy")
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"test-artifact-not-real-weights")
    (checkpoint / "config.yaml").write_text(yaml.safe_dump({"dataloader": {
        "embodiment": "yam_dual", "variant": "real", "action_mode": "eef",
        "num_frames": 33, "video_stride": 4, "normalize_mode": "min-max"}}))
    config = dict(observation_profile="yam_base", env_cfg_type="yam_dual", action_type="ee",
                  ckpt_dir=str(checkpoint), replan_steps=32)
    assert validate_deployment(config)["action_horizon"] == 32
