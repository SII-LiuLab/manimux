"""Shared launcher checks; fixtures do not depend on ignored task folders."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/training"
PREFIXES = ("YAM_", "OPENPI_", "PI05_", "LINGBOT_", "XR1_", "GR00T_", "OPENWAM_")


class TrainingTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = self.root / "storage"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIXES)}
        self.env.update(YAM_TRAIN_ROOT=str(self.storage), NNODES="1", WORLD_SIZE="1", MLP_WORKER_NUM="1")
        self.env.update(
            OPENPI_TASK_NAME="fixture_task", OPENPI_LEROBOT_REPO_ID="fixture_dataset",
            LINGBOT_VLA2_DATASET_PATH=str(self.storage / "datasets/lerobot/fixture"),
            LINGBOT_VLA2_GPU_IDS="0,1,2,3,4,5,6,7",
            XR1_DATASET_PATH=str(self.storage / "datasets/xr1/RoboDojo_real-assemble_the_screwdriver-yam_dual-ee"),
            XR1_DATA_CONFIG_NAME="fixture_xr1", XR1_TASK_NAME="assemble_the_screwdriver",
            XR1_INSTRUCTION="Assemble the screwdriver.", XR1_GPU_IDS="0,1,2,3,4,5,6,7",
            GR00T_TASK_NAME="fixture_task", GR00T_SRC_DATASET="fixture_dataset",
            OPENWAM_DATASET_DIR=str(self.storage / "datasets/openwam/fixture"),
        )

    def run_task(self, name, mode="plan", **env):
        return subprocess.run(
            ["bash", str(SCRIPTS / name), mode, "regression-run"],
            env=dict(self.env, **env), capture_output=True, text=True, timeout=10,
        )

    def touch(self, path, text="fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_shared_launchers_and_example_plan_without_writes(self):
        tasks = sorted(SCRIPTS.glob("train_*_yam_cluster.sh"))
        tasks.append(SCRIPTS / "example/pi05.sh")
        self.assertEqual(len(tasks), 6)
        for script in tasks:
            with self.subTest(script=script.name):
                result = self.run_task(str(script.relative_to(SCRIPTS)))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("effective_global_batch=64", result.stdout)
                self.assertIn("/XPolicyLab/policy/", result.stdout)
                self.assertNotIn("/.local/", result.stdout)
        self.assertFalse(self.storage.exists())

    def test_micro_batch_and_gpu_changes_derive_native_accumulation(self):
        cases = [
            ("train_xr1_yam_cluster.sh", {"XR1_GPU_IDS": "0,1,2,3", "XR1_MICRO_BATCH_SIZE": "2"}, 8),
            ("train_lingbot_vla2_yam_cluster.sh", {"LINGBOT_VLA2_GPU_IDS": "0,1,2,3"}, 16),
            ("train_openwam_yam_cluster.sh", {"OPENWAM_GPU_IDS": "0,1", "OPENWAM_BATCH_SIZE": "2"}, 16),
        ]
        for script, env, accumulation in cases:
            with self.subTest(script=script):
                result = self.run_task(script, **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"accumulation={accumulation}", result.stdout)
                self.assertIn("effective_global_batch=64", result.stdout)
        self.assertFalse(self.storage.exists())

    def test_invalid_batch_fails_before_training_or_preparation(self):
        cases = [
            ("train_pi05_yam_cluster.sh", {"OPENPI_BATCH_SIZE": "32"}),
            ("train_xr1_yam_cluster.sh", {"XR1_GPU_IDS": "0,1,2,3", "XR1_GRAD_ACCUM_STEPS": "8"}),
            ("train_lingbot_vla2_yam_cluster.sh", {"LINGBOT_VLA2_GLOBAL_BATCH_SIZE": "32"}),
            ("train_openwam_yam_cluster.sh", {"OPENWAM_GRADIENT_ACCUMULATION_STEPS": "8"}),
            ("train_gr00t_n17_yam_cluster.sh", {"GR00T_NUM_GPUS": "8"}),
            ("train_xr1_yam_cluster.sh", {"XR1_GPU_IDS": "0,0"}),
            ("train_xr1_yam_cluster.sh", {"XR1_MICRO_BATCH_SIZE": "0"}),
            ("train_xr1_yam_cluster.sh", {"XR1_MICRO_BATCH_SIZE": "3"}),
        ]
        for script, env in cases:
            for mode in ("plan", "prepare", "train"):
                with self.subTest(script=script, env=env, mode=mode):
                    result = self.run_task(script, mode, **env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.storage.exists())

    def test_explicit_historical_global_batch_is_supported(self):
        result = self.run_task(
            "train_pi05_yam_cluster.sh",
            YAM_EXPECTED_GLOBAL_BATCH_SIZE="32", OPENPI_BATCH_SIZE="32",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("effective_global_batch=32", result.stdout)

    def test_xr1_reused_config_cannot_override_requested_micro_batch(self):
        workspace = self.root / "workspace"
        policy = workspace / "XPolicyLab/policy/Xiaomi_Robotics_1"
        dataset = self.storage / "datasets/xr1/RoboDojo_real-assemble_the_screwdriver-yam_dual-ee"
        config = self.touch(dataset / "legacy.yaml", "data:\n  params:\n    train_datasets:\n      batch_size: 16\n")
        for path in (
            dataset / "manifest.json", dataset / "norm_stats.json",
            self.storage / "weights/base/xiaomi/model_states.pt",
            self.storage / "weights/base/xiaomi/qwen3_vl_4b_processor/config.json",
            self.storage / "weights/base/xiaomi/qwen3_vl_4b_processor/tokenizer.json",
            workspace / "scripts/datasets/prepare_xr1_yam_dataset.py",
        ):
            self.touch(path)
        (policy / "xiaomi_robotics_1/xr1/configs/data").mkdir(parents=True)
        fake_python = self.touch(self.storage / "envs/xr1/.venv/bin/python", '''#!/usr/bin/env python3
import os, sys
if len(sys.argv) > 1 and sys.argv[1] == "-":
    if "print(config)" in sys.stdin.read():
        print(os.environ["FIXTURE_CONFIG"])
''')
        fake_python.chmod(0o755)
        capture = self.root / "xr1-argv.json"
        self.touch(policy / "train.sh", '''#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, sys
from pathlib import Path
Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:] + ["RESOURCE_GPU=" + os.environ["RESOURCE_GPU"]]))
PY
''')
        result = self.run_task(
            "train_xr1_yam_cluster.sh", "train",
            XR1_WORKSPACE=str(workspace), XR1_MICRO_BATCH_SIZE="2",
            FIXTURE_CONFIG=str(config), CAPTURE=str(capture), RESOURCE_GPU="4",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(capture.read_text())
        self.assertIn("data.params.train_datasets.batch_size=2", argv)
        self.assertIn("trainer.accumulate_grad_batches=4", argv)
        self.assertIn("RESOURCE_GPU=8", argv)
        self.assertEqual(argv[:6], ["RoboDojo_real", "assemble_the_screwdriver", "yam_dual", "ee", "0", "0,1,2,3,4,5,6,7"])
        self.assertIn("batch_size: 16", config.read_text())

    def test_lingbot_policy_preserves_task_config_and_native_batch_args(self):
        xpl = self.root / "XPolicyLab"
        policy = xpl / "policy/LingBot_VLA2"
        policy.mkdir(parents=True)
        shutil.copyfile(REPO / "XPolicyLab/policy/LingBot_VLA2/train.sh", policy / "train.sh")
        self.touch(xpl / "utils/get_action_dim.sh", "echo 14\n")
        python = self.touch(self.root / "env/bin/python", "#!/bin/sh\nexit 0\n")
        python.chmod(0o755)
        model = self.touch(self.root / "model/model.safetensors.index.json").parent
        dataset = self.touch(self.root / "data/meta/info.json").parent.parent
        stats = self.touch(self.root / "stats.json")
        config = self.touch(self.root / "native-depth.yaml")
        robots = self.touch(self.root / "robots/joint_ee.yaml", "training robot").parent
        deploy = self.touch(self.root / "deploy.yaml", "deployment robot")
        output = self.root / "output"
        capture = self.root / "lingbot-argv.json"
        self.touch(policy / "lingbot_vla_v2/train.sh", '''#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, sys
from pathlib import Path
Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
PY
''')
        env = dict(self.env, LINGBOT_VLA2_ENV_DIR=str(python.parent.parent),
                   LINGBOT_VLA2_MODEL_PATH=str(model), LINGBOT_VLA2_TOKENIZER_PATH=str(model),
                   LINGBOT_VLA2_DATASET_PATH=str(dataset), LINGBOT_VLA2_NORM_STATS_PATH=str(stats),
                   LINGBOT_VLA2_ROBOT_NAME="joint_ee", LINGBOT_VLA2_ROBOT_CONFIG_ROOT=str(robots),
                   LINGBOT_VLA2_TRAINING_CONFIG=str(config), LINGBOT_VLA2_DEPLOY_ROBOT_CONFIG=str(deploy),
                   LINGBOT_VLA2_CHECKPOINT_DIR=str(output), LINGBOT_VLA2_MICRO_BATCH_SIZE="2",
                   LINGBOT_VLA2_GRAD_ACCUM_STEPS="8", LINGBOT_VLA2_GLOBAL_BATCH_SIZE="64",
                   LINGBOT_VLA2_DP_SHARD_SIZE="4", CAPTURE=str(capture))
        result = subprocess.run(["bash", str(policy / "train.sh"), "RoboDojo_real", "task", "yam_dual", "joint", "0", "0,1,2,3"],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(capture.read_text())
        self.assertEqual(argv[1], str(config))
        for key, value in {"--train.micro_batch_size": "2", "--train.gradient_accumulation_steps": "8",
                           "--train.global_batch_size": "64", "--train.data_parallel_shard_size": "4",
                           "--data.robot_config_root": str(robots)}.items():
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertEqual((output / "robot_config.yaml").read_text(), "deployment robot")
        self.assertEqual((output / "training_robot_config.yaml").read_text(), "training robot")


if __name__ == "__main__":
    unittest.main()
