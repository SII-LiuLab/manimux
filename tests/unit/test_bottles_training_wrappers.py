from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "scripts/training"


def test_generic_lingbot_and_xr1_wrappers_are_task_neutral() -> None:
    for name in ("train_lingbot_vla2_yam_cluster.sh", "train_xr1_yam_cluster.sh"):
        source = (TRAINING / name).read_text()
        assert "assemble_the_screwdriver" not in source
        assert "Assemble the screwdriver." not in source

    lingbot = (TRAINING / "train_lingbot_vla2_yam_cluster.sh").read_text()
    assert 'PYTHONPATH="${WORKSPACE_PYTHONPATH}"' in lingbot
    assert "RESOLVED_TRAINING_CONFIG" in lingbot
    assert "--train.align_params" not in lingbot


def test_bottles_wrappers_select_all_data_and_eight_gpus() -> None:
    lingbot = (TRAINING / "train_lingbot_vla2_yam_bottles_cluster.sh").read_text()
    xr1 = (TRAINING / "train_xr1_yam_bottles_cluster.sh").read_text()

    for source in (lingbot, xr1):
        assert "put_bottles_into_the_bin" in source
        assert "0,1,2,3,4,5,6,7" in source
        assert "EXPECTED_EPISODES" in source and "50" in source
        assert "EXPECTED_FRAMES" in source and "35118" in source
        assert "MAX_STEPS" in source and "30000" in source
        assert "5000" in source
