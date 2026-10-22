from pathlib import Path
from manimux.cli import load_config


def test_public_dream_example_is_standalone():
    root = Path(__file__).resolve().parents[2]
    config = load_config(str(root / "manimux/configs/examples/dream1-yam/serial30-example.yaml"))
    assert config["policy"]["worker"] == "dream_api"
    assert config["policy"]["options"]["api_key_env"] == "DREAM_API_KEY"
    assert config["policy"]["horizon_policy_steps"] == 48
    assert config["inference"]["algorithm"] == "serial"
    assert config["inference"]["max_chunk_policy_steps"] == 48
    assert config["robot"]["control_hz"] == 30
    assert config["executor"]["type"] == "direct"
