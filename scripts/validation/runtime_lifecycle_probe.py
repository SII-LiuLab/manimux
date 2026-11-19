"""Hardware-free regression tests for Finish and retained cleanup ownership.

Run from the repository root:
    uv run python -m unittest scripts.validation.runtime_lifecycle_probe -v

The production EdgeRuntime, configuration loader and recorder run normally.
Robot, worker and GUI transports are disconnected in-memory fixtures.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from manimux.cli import load_config
from manimux.dispatch.demo import DemoRobot
from manimux.embodiments.arm.tianji.arm import TianjiArmSettings, TianjiController
from manimux.embodiments.robot.base import RobotBase
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.fake import FakePolicyAdapter
from manimux.robogui.dashboard import PolicyRoboGUI
from manimux.robogui.publisher import RoboGUIBridge, RoboGUIControl
from manimux.runtime import build_runtime, inference_parameters
from manimux.runtime.dispatch import PolicyDispatch
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.inference import INFERENCE_ALGORITHMS
from manimux.runtime.lifecycle import ShutdownStatus
from manimux.session import RuntimeSessionService

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "manimux/configs/experiments/example-synthetic/demo/serial.yaml"


class ProbeAbort(BaseException):
    """Fail a bounded probe without being swallowed by production recovery loops."""


class MemoryWorker:
    capabilities = PolicyCapabilities(frozenset({"default", "rtc"}))
    is_alive = True

    def __init__(self, *args, **kwargs):
        self.stopped = False

    def start(self):
        pass

    def request_stop(self):
        self.stopped = True
        self.is_alive = False

    def close(self):
        self.request_stop()


class MemoryBridge(RoboGUIBridge):
    def __init__(self):
        super().__init__(enabled=False, robot="synthetic")
        self.control = RoboGUIControl(paused=True, finish_requested=True)
        self.events = []

    def poll_control(self):
        return replace(
            self.control,
            episode_id=self._state_metadata["episode_id"],
            control_run_dir=self._state_metadata["run_dir"],
        )

    def publish_event(self, event, **kwargs):
        self.events.append(event)


class LifecycleRobot(DemoRobot):
    home_before_close = False

    def __init__(self, config, clock):
        super().__init__(config, clock)
        self.calls = []
        self.close_failures = 0

    def connect(self):
        self.calls.append("connect")

    def home(self):
        self.calls.append("home")

    def close(self):
        self.calls.append("close")
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("fixture close failed; controller still owned")


class RuntimeLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name)
        self.config = load_config(EXPERIMENT)
        self.config["run"]["warmup_before_start"] = False
        self.config["robot"]["options"]["execute"] = True
        self.config["run"]["max_control_steps"] = 1
        self.config["run"]["sensor_reading"]["mode"] = "inline"
        self.bridge = MemoryBridge()
        self.robots = []
        self.retry_calls = 0
        self.on_retry = lambda: None
        self.status = None
        self.robot_factory = LifecycleRobot

        def build_robot(config, clock):
            robot = self.robot_factory(config, clock)
            self.robots.append(robot)
            return robot

        def poll_cleanup():
            self.retry_calls += 1
            if self.retry_calls > 10:
                raise ProbeAbort("cleanup retry did not converge")
            try:
                self.on_retry()
            except Exception as exc:
                raise ProbeAbort(str(exc)) from exc
            return {
                "shutdown_recovery": {
                    **self.status.identity,
                    "token": self.status._recovery_token,
                    "action": "retry_close",
                }
            }

        def build_status(**kwargs):
            self.status = ShutdownStatus(
                **kwargs,
                publisher_factory=Mock,
                control_factory=lambda: Mock(poll_state=poll_cleanup),
            )
            return self.status

        for target, replacement in (
            ("build_robot", build_robot),
            ("PolicyWorkerClient", MemoryWorker),
            ("RoboGUIBridge", lambda **kwargs: self.bridge),
            ("OperationStatus", Mock),
            ("ShutdownStatus", build_status),
        ):
            patcher = patch(f"manimux.runtime.edge.{target}", replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def runtime(self, *, home=None, mandatory_home=False):
        self.bridge.control = replace(self.bridge.control, finish_home=home)
        runtime = build_runtime(self.config, self.run_dir)
        runtime._robot.home_before_close = mandatory_home
        return runtime

    def test_finish_without_home_closes_at_current_pose(self):
        runtime = self.runtime(home=False)
        runtime.run()
        self.assertEqual(runtime._robot.calls, ["connect", "close"])

    def test_finish_with_home_keeps_existing_order(self):
        runtime = self.runtime(home=True)
        runtime.run()
        self.assertEqual(runtime._robot.calls, ["connect", "home", "close"])

    def test_legacy_finish_keeps_default_home(self):
        runtime = self.runtime()
        runtime.run()
        self.assertEqual(runtime._robot.calls, ["connect", "home", "close"])

    def test_mandatory_home_cannot_be_bypassed(self):
        runtime = self.runtime(home=False, mandatory_home=True)
        runtime.run()
        self.assertEqual(runtime._robot.calls, ["connect", "home", "close"])

    def test_assembly_without_home_only_closes(self):
        runtime = self.runtime(home=True)
        runtime._robot.home = None
        runtime.run()
        self.assertEqual(runtime._robot.calls, ["connect", "close"])

    def test_registered_strategies_share_finish_home_contract(self):
        # Exercise the actual registered strategy startup/exit paths, without
        # making any inference request or claiming model/algorithm validation.
        base = deepcopy(self.config)
        model = {
            "policy_family": "pi05",
            "action_type": "joint",
            "observation_profile": "yam_native",
            "execution": {"backend": "jax"},
            "action_horizon": 50,
            "action_dim": 14,
            "num_steps": 10,
            "inference_seed": 0,
            "denoise_trace": {
                "enabled": True,
                "schema_version": 1,
                "field": "v",
                "axes": ["denoising_step", "action_step", "model_action"],
                "coordinate_space": "normalized_model",
                "velocity_kind": "raw_network_velocity",
                "dtype": "float32",
                "model_action_dim": 14,
                "valid_action_indices": list(range(14)),
                "supported_modes": ["default", "rtc"],
            },
            "rtc_guidance": {
                "padding_mask": "valid_action_indices",
                "valid_action_indices": list(range(14)),
            },
        }
        capabilities = PolicyCapabilities(
            frozenset({"default", "rtc", "train_time_rtc", "aac", "paint", "autohorizon"}),
            {"model": model},
        )
        with (
            patch.object(MemoryWorker, "capabilities", capabilities),
            patch.object(FakePolicyAdapter, "supports_candidate_decoding", True),
        ):
            for algorithm in sorted(INFERENCE_ALGORITHMS):
                for home in (False, True):
                    with self.subTest(algorithm=algorithm, home=home):
                        self.config = deepcopy(base)
                        options = {"blend_policy_steps": 0}
                        if algorithm == "aac":
                            options["aac"] = {
                                "ee_stats_path": str(
                                    ROOT / "scripts/validation/starvla_fixtures/aac_stats.json"
                                )
                            }
                        if algorithm == "bid_backward":
                            options["action_start_mode"] = "first_step_when_ready"
                        if algorithm == "train_time_rtc":
                            options["action_start_mode"] = "drop_infer_latency"
                        self.config["inference"] = inference_parameters(
                            executor=self.config["executor"],
                            algorithm=algorithm,
                            **options,
                        )
                        if algorithm.startswith("chunktrust"):
                            self.config["policy"]["horizon_policy_steps"] = 50
                            self.config["policy"]["worker"] = "xpolicylab_ws"
                        runtime = self.runtime(home=home)
                        runtime.run()
                        expected = ["connect", "home", "close"] if home else ["connect", "close"]
                        self.assertEqual(runtime._robot.calls, expected)

    def test_partial_connection_failure_still_retries_close(self):
        runtime = self.runtime(home=False)
        runtime._robot.connect = Mock(side_effect=RuntimeError("partial connection"))
        runtime._robot.close_failures = 1
        with self.assertRaisesRegex(ExceptionGroup, "runtime cleanup failed"):
            runtime.run()
        self.assertEqual(self.retry_calls, 1)
        self.assertEqual(runtime._robot.calls, ["close", "close"])

    def test_dispatch_finish_uses_same_lifecycle_with_and_without_policy(self):
        for policy_enabled in (False, True):
            with self.subTest(policy_enabled=policy_enabled):
                dispatch = PolicyDispatch(self.config, policy_enabled=policy_enabled)
                self.bridge.control = replace(self.bridge.control, finish_home=False)
                runtime = EdgeRuntime(self.config, self.run_dir, dispatch=dispatch)
                runtime.run()
                self.assertEqual(runtime._robot.calls, ["connect", "close"])
                self.assertTrue(dispatch.status()["closed"])

    def test_fault_and_interrupt_keep_cleanup_owned_until_retry(self):
        for error in (RuntimeError("lost feedback"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                runtime = self.runtime()
                state = runtime._robot.get_state()
                runtime._robot.get_state = Mock(side_effect=[state, error])
                runtime._robot.close_failures = 1
                retries_before = self.retry_calls
                with self.assertRaisesRegex(ExceptionGroup, "runtime cleanup failed"):
                    runtime.run()
                self.assertEqual(self.retry_calls, retries_before + 1)
                self.assertEqual(runtime._robot.calls, ["connect", "home", "close", "close"])

    def test_failed_close_retries_original_robot_without_repeating_home(self):
        runtime = self.runtime(home=True)
        runtime._robot.close_failures = 2

        def while_blocked():
            self.assertEqual(len(self.robots), 1)
            self.assertNotIn("episode_finished", self.bridge.events)
            self.assertEqual(self.status._phase, "cleanup_blocked")
            self.assertTrue(runtime._worker.stopped)

        self.on_retry = while_blocked
        with self.assertRaisesRegex(ExceptionGroup, "runtime cleanup failed"):
            runtime.run()
        self.assertEqual(self.retry_calls, 2)
        self.assertEqual(runtime._robot.calls, ["connect", "home", "close", "close", "close"])
        self.assertNotIn("episode_finished", self.bridge.events)
        results = list(self.run_dir.rglob("result.json"))
        self.assertEqual(len(results), 1)
        self.assertFalse(json.loads(results[0].read_text())["run_completed"])

    def test_stale_feedback_retains_sdk_owner_until_session_cleanup_retries(self):
        self._check_sdk_cleanup_before_next_prepare("feedback")

    def test_failed_sdk_release_blocks_next_prepare_until_original_owner_retries(self):
        self._check_sdk_cleanup_before_next_prepare("release")

    def _check_sdk_cleanup_before_next_prepare(self, failure):
        fault = {"active": True}
        controllers = []
        settings = {
            "left": TianjiArmSettings(
                joint_limits=(-np.ones(7), np.ones(7)),
                velocity_ratio=10,
                acceleration_ratio=10,
            )
        }

        class OwnedAssembly(RobotBase):
            def __init__(self, controller, model):
                super().__init__(
                    arm_components={
                        "left_arm": SimpleNamespace(
                            controller=controller,
                            channel="left",
                            num_joints=7,
                        )
                    },
                    models={"left_arm": model},
                )

        class OwnedRobot(LifecycleRobot):
            def __init__(self, config, clock):
                super().__init__(config, clock)
                self.controller = controller = TianjiController(settings=settings)
                self.assembly = OwnedAssembly(controller, self.kinematics.models["left_arm"])
                controllers.append(controller)

                def read(**kwargs):
                    if failure == "feedback" and fault["active"]:
                        raise RuntimeError("stale feedback")
                    return {}, {"left": 0}, {}

                controller._read = read

            def connect(self):
                super().connect()
                self.controller._acquire_session()
                self.controller._owns_session = True
                self.controller._enabled = {"left"}
                self.controller._robot = Mock(
                    release_robot=lambda: not (failure == "release" and fault["active"])
                )
                self.assembly._controller_open = [self.controller]

            def close(self):
                self.calls.append("close")
                self.assembly.close()

        self.robot_factory = OwnedRobot
        self.bridge.control = replace(self.bridge.control, finish_home=False)
        factory = Mock(side_effect=lambda config, path: build_runtime(config, path))
        with (
            patch("manimux.embodiments.arm.tianji.arm._CONTROL_SESSION", threading.Lock()),
            patch("manimux.embodiments.arm.tianji.arm._pending_clear_session", None),
            patch("manimux.session._IdleRecovery", return_value=Mock(available=False)),
            patch("manimux.runtime.lifecycle.OperationStatus", Mock),
        ):
            service = RuntimeSessionService(
                self.config,
                self.run_dir / failure,
                runtime_factory=factory,
                publisher_factory=Mock,
            )
            service._wait_for_rollout_request = Mock(return_value={})

            def while_blocked():
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(service._wait_for_rollout_request.call_count, 1)
                self.assertTrue(controllers[0]._owns_session)
                with self.assertRaisesRegex(RuntimeError, "another TianjiController owns"):
                    TianjiController(settings=settings)._acquire_session()
                fault["active"] = False

            self.on_retry = while_blocked
            try:
                service.serve(max_rollout_attempts=2)
                self.assertEqual(factory.call_count, 2)
                self.assertEqual(len(controllers), 2)
                self.assertTrue(all(not owner._owns_session for owner in controllers))
            finally:
                fault["active"] = False
                for owner in controllers:
                    owner.close()


class CleanupRequestTests(unittest.TestCase):
    def test_only_current_retry_request_advances_cleanup(self):
        home = Mock()
        status = ShutdownStatus(
            enabled=True,
            robot="synthetic",
            policy="fake",
            run_dir="fixture",
            episode_id="1",
            publisher_factory=Mock,
            home=home,
        )
        self.addCleanup(status.close)
        requests = iter(
            (
                {"token": "stale", "action": "retry_close"},
                {"episode_id": "other", "action": "retry_close"},
                {"run_dir": "other", "action": "retry_close"},
                {"action": "release", "supported": True},
                {"action": "home", "path_clear": True},
                {"action": "retry_close"},
            )
        )
        seen = []

        def poll():
            if len(seen) >= 6:
                raise ProbeAbort("current cleanup request was not accepted")
            request = {
                **status.identity,
                "token": status._recovery_token,
                **next(requests),
            }
            seen.append(request)
            return {"shutdown_recovery": request}

        client = Mock(poll_state=poll)
        status._control_factory = lambda: client
        with patch("manimux.runtime.lifecycle.time.sleep"):
            status.wait_for_cleanup("close failed")
        self.assertEqual(len(seen), 6)
        home.assert_not_called()
        self.assertIsNone(status._home)
        client.close.assert_called_once()
        self.assertEqual(status._phase, "closing")
        self.assertEqual(status._recovery_token, "")

    def test_transport_failure_and_interrupt_keep_owner_waiting(self):
        status = ShutdownStatus(
            enabled=True,
            robot="synthetic",
            policy="fake",
            run_dir="fixture",
            episode_id="1",
            publisher_factory=Mock,
        )
        self.addCleanup(status.close)
        tokens = []

        def client_factory():
            tokens.append(status._recovery_token)
            if len(tokens) == 1:
                raise OSError("GUI unavailable")
            if len(tokens) == 2:
                return Mock(poll_state=Mock(side_effect=KeyboardInterrupt))
            return Mock(
                poll_state=lambda: {
                    "shutdown_recovery": {
                        **status.identity,
                        "token": status._recovery_token,
                        "action": "retry_close",
                    }
                }
            )

        status._control_factory = client_factory
        with patch("manimux.runtime.lifecycle.time.sleep"):
            status.wait_for_cleanup("close failed")
        self.assertEqual(len(set(tokens)), 3)
        self.assertEqual(status._phase, "closing")


class DashboardCleanupTests(unittest.TestCase):
    def setUp(self):
        self.gui = gui = PolicyRoboGUI.__new__(PolicyRoboGUI)
        gui.lock = threading.RLock()
        gui.shutdown_phase = "cleanup_blocked"
        gui._shutdown_recovery_token = "current"
        gui._shutdown_recovery_identity = {"run_dir": "fixture", "episode_id": "1"}
        gui._shutdown_recovery_request = {}
        gui._shutdown_can_home = True
        gui.last_shutdown_time = time.time()
        for field in (
            "shutdown_path_clear",
            "shutdown_supported",
            "shutdown_home_btn",
            "shutdown_release_btn",
            "shutdown_retry_btn",
            "shutdown_recovery_text",
        ):
            setattr(gui, field, SimpleNamespace(disabled=False, visible=True, value=True))

    def test_cleanup_retry_does_not_offer_home_or_release_override(self):
        self.gui._sync_shutdown_recovery_buttons()
        self.assertFalse(self.gui.shutdown_retry_btn.disabled)
        self.assertTrue(self.gui.shutdown_home_btn.disabled)
        self.assertTrue(self.gui.shutdown_release_btn.disabled)
        self.assertFalse(self.gui.shutdown_supported.visible)
        self.gui._request_shutdown_recovery("retry_close")
        request = self.gui._shutdown_recovery_request
        self.assertEqual(request["action"], "retry_close")
        self.assertEqual(request["token"], "current")
        self.assertEqual(request["episode_id"], "1")
        self.assertTrue(self.gui.shutdown_retry_btn.disabled)

    def test_cleanup_heartbeat_blocks_prepare_and_exposes_retry(self):
        gui = self.gui
        gui.service_id = "fixture"
        gui._episode_id = "1"
        gui.preparing_rollout = gui.episode_finalized = False
        gui.service_ready = gui.episode_active = True
        gui._shutdown_detail = ""
        for field in (
            "camera_view",
            "_set_stage",
            "_set_setup_controls_enabled",
            "_set_policy_controls_enabled",
            "_set_evaluation_enabled",
            "_report_error",
            "status",
            "executor_info",
            "shutdown_recovery_folder",
        ):
            setattr(gui, field, Mock())
        gui._update_event(
            {
                "event": "runtime_shutdown",
                "metadata": {
                    "run_dir": "fixture",
                    "episode_id": "1",
                    "phase": "cleanup_blocked",
                    "recovery_token": "fresh",
                    "detail": "SDK release failed",
                    "can_home": True,
                },
            }
        )
        self.assertFalse(gui.service_ready)
        self.assertFalse(gui.episode_active)
        gui._set_setup_controls_enabled.assert_called_once_with(False)
        self.assertFalse(gui.shutdown_retry_btn.disabled)
        self.assertTrue(gui.shutdown_home_btn.disabled)
        self.assertEqual(gui._shutdown_recovery_token, "fresh")

    def test_stale_cleanup_heartbeat_disables_retry(self):
        self.gui.last_shutdown_time = time.time() - 4
        self.gui._request_shutdown_recovery("retry_close")
        self.assertEqual(self.gui._shutdown_recovery_request, {})

    def test_supported_release_gate_retains_its_separate_confirmations(self):
        self.gui.shutdown_phase = "release_blocked"
        self.gui._sync_shutdown_recovery_buttons()
        self.assertTrue(self.gui.shutdown_retry_btn.disabled)
        self.assertFalse(self.gui.shutdown_home_btn.disabled)
        self.assertFalse(self.gui.shutdown_release_btn.disabled)


if __name__ == "__main__":
    unittest.main()
