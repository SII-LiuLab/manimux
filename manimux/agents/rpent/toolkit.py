"""RPent-native tools backed by ManiMux; import only in the planner environment."""

from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from PIL import Image
from pydantic import Field
from rpent.dashboard.events import DashboardEventSink, NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Tool, ToolResult, tool
from rpent.tools.toolkit import Toolkit

from manimux.dispatch.client import DispatchClient


class ManiMuxToolkit(Toolkit):
    """Observe and schedule an existing policy; ManiMux owns devices and rollouts."""

    def __init__(
        self,
        client: DispatchClient,
        *,
        output_dir: Path,
        memory: MemoryManager,
        dashboard_events: DashboardEventSink | None = None,
        planner_label: str = "RPent tools",
        operator_control: bool = False,
    ) -> None:
        super().__init__(
            memory=memory,
            state=EnvState(output_dir),
            dashboard_events=dashboard_events or NullDashboardEventSink(),
        )
        self.client = client
        self._operator_control = operator_control
        status = self.client.status()
        self.policy_enabled = status.get("policy_enabled", True)
        if type(self.policy_enabled) is not bool:
            raise ValueError("ManiMux status policy_enabled must be a boolean")
        self.execute_policy = None
        self.policy_contract = None
        if self.policy_enabled:
            horizon = status.get("horizon_policy_steps")
            if type(horizon) is not int or horizon < 1:
                raise ValueError("ManiMux status requires a positive integer horizon_policy_steps")
            # Each rollout binds native validation to its selected policy horizon.
            self.execute_policy = self._policy_tool(horizon)
            self.policy_contract = {
                "horizon_policy_steps": horizon,
                "use_length": self.execute_policy.input_schema["properties"]["use_length"],
            }
            self.add_tool(self.execute_policy)
        self.add_tool(self.observe)
        self._report_decisions = bool(status.get("agent_decision_telemetry"))
        self._planner_label = planner_label
        motion = status.get("motion")
        self.motion_description = motion
        if motion is not None:
            for method in (self.move_to, self.move_delta, self.rotate, self.set_gripper):
                self.add_tool(method)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Observe the native RPent tool boundary without choosing any actions."""
        self._wait_for_operator()
        if not self._report_decisions or name not in {
            "observe", "move_to", "move_delta", "rotate", "set_gripper", "execute_policy", "finish"
        }:
            return super().execute_tool(name, input_dict)
        decision = dict(decision_id=uuid.uuid4().hex, tool=name, arguments=input_dict,
                        planner=self._planner_label)
        def report(phase: str, outcome: dict[str, Any]) -> None:
            try:
                self.client.report_agent_decision(**decision, phase=phase, outcome=outcome)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Could not publish agent decision", exc_info=True
                )
        started = time.monotonic()
        report("selected", {})
        try:
            result = super().execute_tool(name, input_dict)
        except BaseException as exc:
            report("failed", {"error": str(exc)[:1000], "elapsed_s": time.monotonic() - started})
            raise
        outcome = {key: result.data[key] for key in (
            "request_id", "state", "reason", "error", "arrival", "completed_chunks",
            "status", "summary",
        ) if key in result.data}
        outcome["elapsed_s"] = time.monotonic() - started
        report("failed" if result.is_error else "completed", outcome)
        return result

    def _wait_for_operator(self) -> None:
        if not self._operator_control:
            return
        while True:
            self.raise_if_cancelled()
            status = self.client.status()
            if status["closed"] or not status["ready"]:
                raise RuntimeError("ManiMux rollout ended or is unavailable")
            if status["enabled"]:
                return
            time.sleep(0.1)

    @staticmethod
    def _observation_result(observation: dict[str, Any]) -> ToolResult:
        frames = observation.get("frames", {})
        return ToolResult(
            data={
                **{k: v for k, v in observation.items() if k != "frames"},
                "camera_names": list(frames),
            },
            images=[base64.b64decode(frame["data"], validate=True) for frame in frames.values()],
        )

    @tool(readonly=True)
    def observe(self) -> ToolResult:
        """Read current robot state and RGB views from the active ManiMux rollout."""
        result = self._observation_result(self.client.observe())
        result.data["motion_contract"] = self.motion_description
        if self.policy_enabled:
            result.data["policy_contract"] = deepcopy(self.policy_contract)
        return result

    def _motion(self, name: str, arguments: dict[str, Any], timeout_s: float) -> ToolResult:
        result = self.client.motion(
            name, arguments, timeout_s=timeout_s, check_cancelled=self.raise_if_cancelled
        )
        if result["state"] != "completed":
            result["error"] = result["reason"] or result["state"]
        return ToolResult(data=result)

    @tool
    def move_to(
        self,
        group: str,
        xyz: list[float],
        quaternion_wxyz: list[float] | None = None,
        timeout_s: Annotated[float, Field(gt=0, le=3600)] = 120.0,
    ) -> ToolResult:
        """Move the TCP to XYZ metres in the group's base frame from observe.

        Optional orientation is a unit wxyz quaternion; omission preserves measured
        orientation. Uses bounded joint interpolation, not a Cartesian straight line.
        """
        return self._motion(
            "move_to", dict(group=group, xyz=xyz, quaternion_wxyz=quaternion_wxyz), timeout_s
        )

    @tool
    def move_delta(
        self,
        group: str,
        delta_xyz: list[float],
        timeout_s: Annotated[float, Field(gt=0, le=3600)] = 120.0,
    ) -> ToolResult:
        """Translate from measured TCP by XYZ metres in the group's base frame."""
        return self._motion("move_delta", dict(group=group, delta_xyz=delta_xyz), timeout_s)

    @tool
    def rotate(
        self,
        group: str,
        delta_rpy: list[float],
        timeout_s: Annotated[float, Field(gt=0, le=3600)] = 120.0,
    ) -> ToolResult:
        """Apply XYZ Euler increments in radians about the group's base axes."""
        return self._motion("rotate", dict(group=group, delta_rpy=delta_rpy), timeout_s)

    @tool
    def set_gripper(
        self,
        group: str,
        opening: Annotated[float, Field(ge=0, le=1)],
        timeout_s: Annotated[float, Field(gt=0, le=3600)] = 120.0,
    ) -> ToolResult:
        """Set normalized opening: 0 closed, 1 open; return measured completion."""
        return self._motion("set_gripper", dict(group=group, opening=opening), timeout_s)

    def _policy_tool(self, horizon: int) -> Tool:
        """Declare the native tool from the active runtime, without upstream patches."""
        default_length = min(5, horizon)

        @tool
        def execute_policy(
            prompt: str,
            chunks: Annotated[int, Field(ge=1, le=1000)] = 1,
            use_length: Annotated[int, Field(
                ge=1, le=horizon,
                description=f"Policy rows retained per chunk, an integer in 1..{horizon} "
                            "for the selected ManiMux experiment.",
                json_schema_extra={"default": default_length},
            )] = default_length,
            timeout_s: Annotated[float, Field(gt=0, le=3600)] = 120.0,
        ) -> ToolResult:
            """Run the configured VLA for bounded chunks using a subtask prompt.

            Respect the selected policy's use_length bound in the tool schema and
            observe.policy_contract. Completion reports dispatched references, not
            physical task success. Inspect the returned observation before deciding
            whether to retry or finish.
            """
            result = self.client.execute(
                prompt,
                chunks=chunks,
                use_length=use_length,
                timeout_s=timeout_s,
                check_cancelled=self.raise_if_cancelled,
            )
            if result["state"] != "completed":
                result["error"] = result["reason"] or result["state"]
            return ToolResult(data=result)

        return execute_policy

    def get_env_state(
        self, *, command: dict[str, Any], result: dict[str, Any], elapsed_s: float
    ) -> ToolResult:
        result = dict(result)
        if self._operator_control and result.get("reason") == "operator_paused":
            self._wait_for_operator()
            result["observation"] = self.client.observe()
        observation = result.pop("observation", None)
        if observation is None:
            observation = self.client.observe()
        with self.state.record_step(
            state=observation["state"],
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"manimux_session_id": self.client.session_id},
        ):
            self.state.save("dispatch_result.json", result)
            for index, frame in enumerate(observation.get("frames", {}).values()):
                image = Image.open(io.BytesIO(base64.b64decode(frame["data"], validate=True)))
                self.state.save(f"camera_{index}.png", np.asarray(image))
        output = self._observation_result(observation)
        output.data = {"observation": output.data, **result}
        return output

    def solved(self) -> bool:
        # A completed command budget is not an operator/evaluator task verdict.
        return False
