"""Legacy native RPC facade retained for protocol regression checks only."""

from __future__ import annotations

import argparse
from typing import Any


def build_facade(environment: Any) -> Any:
    from rpent.robots.components.env_facade_base import BaseEnvFacade

    class Facade(BaseEnvFacade):  # type: ignore[misc]  # Optional, externally typed dependency.
        def _register_rpc(self) -> None:
            super()._register_rpc()
            for name in getattr(
                environment,
                "rpc_methods",
                (
                    "observe",
                    "read_control_state",
                    "control_step",
                    "is_started",
                    "plan_arm_path",
                ),
            ):
                self._rpc[f"env.{name}"] = getattr(environment, name)
            self._rpc["env.request_stop"] = self.request_stop
            for name in (
                "reset_to_configured_qpos",
                "recover_up",
                "release_gripper",
            ):
                self._rpc[f"env.{name}"] = self.unsupported

        @staticmethod
        def unsupported(*args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "Operator recovery is not an agent primitive; use ManiMux idle recovery"
            )

        def _dispatch(
            self,
            method: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            session_id: str | None = None,
        ) -> Any:
            if method == "env.request_stop":
                return self.request_stop()
            if method == "shutdown":
                environment.close()
                self._shutdown_event.set()
                return {"ok": True}
            # The environment serializes admission/reset, while the control
            # thread owns writes. Never hold the facade lock across a batch:
            # observation and a second (rejected) submission must remain prompt.
            if method in self._rpc:
                return self._rpc[method](*args, **kwargs)
            return super()._dispatch(method, args, kwargs, session_id=session_id)

        def request_stop(self) -> dict[str, Any]:
            environment.request_stop()
            return {"stop_requested": True, "hold_confirmed": False}

        def get_env_meta(self) -> dict[str, Any]:
            return environment.get_env_meta()

        def get_action_spec(self) -> None:
            raise NotImplementedError("direct-action discovery is not supported")

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            return environment.reset(*args, **kwargs)

        def step(self, *args: Any, **kwargs: Any) -> Any:
            return environment.step(*args, **kwargs)

        def chunk_step(self, *args: Any, **kwargs: Any) -> Any:
            return environment.chunk_step(*args, **kwargs)

        def get_task_language(self) -> str:
            return environment.get_task_language()

        def get_camera_meta(self, *args: Any, **kwargs: Any) -> Any:
            return environment.get_camera_meta(*args, **kwargs)

        def render_camera(self, *args: Any, **kwargs: Any) -> Any:
            return environment.render_camera(*args, **kwargs)

        def close(self) -> None:
            environment.close()

    return Facade()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retired native RPent environment launcher")
    parser.error(
        "The separate native environment runtime was retired. Use "
        "manimux-rpent serve --config <agent.yaml>, then manimux-rpent run "
        "with the same recipe. Motion tools and execute_policy share that runtime; "
        "select the VLA in the ManiMux experiment. See the RPent guide for migration."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
