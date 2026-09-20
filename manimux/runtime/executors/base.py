from __future__ import annotations

from typing import Protocol

from manimux.types import ActionHorizon, RobotCommand, RobotState


class ExecutorError(RuntimeError):
    pass


class Executor(Protocol):
    @property
    def horizon_steps(self) -> int: ...

    def reset(self, state: RobotState) -> None: ...

    def step(
        self,
        now_ns: int,
        state: RobotState,
        reference: ActionHorizon,
    ) -> RobotCommand: ...
