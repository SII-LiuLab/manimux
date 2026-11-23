"""Manage the native RPent process inside the ordinary RoboGUI rollout lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import uuid
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

import yaml

from manimux.agents.execution import agent_parameters
from manimux.agents.rpent.cli import load_agent_config
from manimux.agents.rpent.transcript import TranscriptFollower
from manimux.dispatch.server import DispatchServer
from manimux.runtime.dispatch import PolicyDispatch
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.motion import MotionTools


class ManagedRPentRuntime:
    """Only EdgeRuntime owns devices. The planner starts after acknowledged Start."""

    def __init__(self, config: dict, run_dir: Path, *, launch_mode: str = "serve"):
        self.config = deepcopy(config)
        self.run_dir = Path(run_dir)
        self.launch = agent_parameters(self.config.get("agent") or {})
        self.recipe_path = Path(self.launch["config"])
        self.recipe = load_agent_config(self.recipe_path, {})
        if "policy_enabled" in self.launch:
            self.recipe = self.recipe.model_copy(update={
                "policy_enabled": self.launch["policy_enabled"],
            })
        if not self.recipe.policy_enabled:
            self.config["agent"]["policy_enabled"] = False
            self.config["run"]["warmup_before_start"] = False
        if self.recipe.motion is None:
            raise ValueError("RPent GUI execution requires a motion profile")
        self.dispatch = PolicyDispatch(self.config, policy_enabled=self.recipe.policy_enabled)
        options = yaml.safe_load(self.recipe.motion.read_text())
        if not isinstance(options, dict):
            raise ValueError("RPent motion profile must be a mapping")
        guard = options.get("path_validator", {}).get("options", {})
        if "extrinsics_path" in guard:
            guard["extrinsics_path"] = str(
                (self.recipe.motion.parent / Path(guard["extrinsics_path"]).expanduser()).resolve()
            )
        for path in (self.recipe_path, self.recipe.motion, guard.get("extrinsics_path")):
            if path is not None:
                source = Path(path).resolve()
                self.config.setdefault("config_sources", {})[str(source)] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
        self.environment = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[3])
        self.environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            source_root, self.environment.get("PYTHONPATH", "")
        )))
        self.environment["PYTHONUNBUFFERED"] = "1"
        # Import validation has no provider request or device connection.
        check = subprocess.run(
            [self.launch["python"], "-c", "import manimux.agents.rpent.toolkit"],
            env=self.environment, capture_output=True, text=True, timeout=20,
        )
        if check.returncode:
            raise ValueError("RPent planner environment is unavailable: " + check.stderr[-1500:])
        self.artifacts = self.run_dir / "agents" / f"rpent-{uuid.uuid4().hex}"
        self.artifacts.mkdir(parents=True)
        self.config["policy"]["options"]["response_log_dir"] = str(
            self.artifacts / "provider-responses"
        )
        self.config["run"]["agent_run_dir"] = str(self.artifacts.resolve())
        (self.artifacts / "runtime-config.json").write_text(
            json.dumps(self.config, indent=2, default=str) + "\n"
        )
        (self.artifacts / "motion.yaml").write_text(yaml.safe_dump(options, sort_keys=False))
        self.runtime = EdgeRuntime(self.config, self.run_dir, dispatch=self.dispatch,
                                   launch_mode=launch_mode)
        self.dispatch.motion_tools = MotionTools(
            self.runtime._robot.kinematics, options, self.config["robot"]["control_hz"]
        )
        self.stop = threading.Event()
        self.process: subprocess.Popen | None = None
        self.error = ""
        self.log = None

    def _start_planner(self, endpoint: str) -> None:
        recipe = self.recipe.model_copy(update={
            "endpoint": endpoint,
            "task": self.config["run"]["task_prompt"],
            "output_dir": self.artifacts / "planner",
            "operator_control": True,
        })
        path = self.artifacts / "agent.yaml"
        path.write_text(yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False))
        self.log = (self.artifacts / "planner.log").open("w")
        self.process = subprocess.Popen(
            [self.launch["python"], "-m", "manimux.agents.rpent", "run", "--config", str(path)],
            cwd=self.artifacts, env=self.environment, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )

    def _supervise(self, endpoint: str) -> None:
        phase = "waiting"
        self.dispatch.set_agent_status(phase, detail="Start rollout to begin RPent decisions.")
        transcript, ticks = TranscriptFollower(self.artifacts / "planner"), 0
        try:
            while not self.stop.wait(0.05):
                ticks += 1
                if self.process is not None and ticks % 5 == 0:
                    self.dispatch.add_agent_messages(transcript.poll())
                status = self.dispatch.status()
                if status["closed"] or self.dispatch.stop_requested:
                    return
                if self.process is None:
                    if not (status["ready"] and status["enabled"]):
                        continue
                    self._start_planner(endpoint)
                    phase = "running"
                    self.dispatch.set_agent_status(phase)
                elif self.process.poll() is not None:
                    result_path = self.artifacts / "planner" / "planner_result.json"
                    result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    if (self.process.returncode or result.get("error")
                            or result.get("finish_result") is None
                            or result.get("manimux_session_id") != self.dispatch.session_id):
                        raise RuntimeError(
                            "RPent exited without a valid verdict. See "
                            f"{self.artifacts / 'planner.log'}"
                        )
                    self.dispatch.set_agent_status("completed", verdict=result["finish_result"])
                    self.dispatch.finish(session_id=self.dispatch.session_id)
                    return
                else:
                    current = "running" if status["enabled"] else "paused"
                    if current != phase:
                        phase = current
                        self.dispatch.set_agent_status(phase)
        except Exception as exc:
            self.error = str(exc)
            self.dispatch.set_agent_status("failed", detail=self.error)
            self.dispatch.finish(session_id=self.dispatch.session_id)
        finally:
            self._stop_planner()
            self.dispatch.add_agent_messages(transcript.poll(final=True))

    def _stop_planner(self) -> None:
        if self.process is not None:
            # Terminate this rollout's process group, including provider children.
            # Robot stop/Home/close remain exclusively in EdgeRuntime.
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=2)
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=2)
        if self.log is not None:
            self.log.close()

    def run(self):
        # A new loopback endpoint and session prevent prior planners reaching the next rollout.
        with DispatchServer(self.dispatch, port=0,
                            token=os.environ.get("MANIMUX_DISPATCH_TOKEN")) as server:
            endpoint = f"http://127.0.0.1:{server.port}"
            supervisor = threading.Thread(target=self._supervise, args=(endpoint,),
                                          name="rpent-supervisor", daemon=True)
            supervisor.start()
            try:
                result = self.runtime.run()
            finally:
                self.dispatch.close("runtime_stopped")
                self.stop.set()
                supervisor.join()
            if self.error:
                raise RuntimeError(self.error)
            return result
