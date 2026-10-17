"""Serve RPent's YAM VLA environment using the ManiMux runtime."""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manimux.integrations.rpent.environment import YamEnvironment


def build_facade(environment: YamEnvironment) -> Any:
    from rpent.robots.components.env_facade_base import BaseEnvFacade

    class Facade(BaseEnvFacade):  # type: ignore[misc]  # Optional, externally typed dependency.
        def _register_rpc(self) -> None:
            super()._register_rpc()
            for name in ("observe", "read_control_state", "control_step", "is_started"):
                self._rpc[f"env.{name}"] = getattr(environment, name)
            self._rpc["env.request_stop"] = self.request_stop
            for name in (
                "plan_arm_path",
                "reset_to_configured_qpos",
                "recover_up",
                "release_gripper",
            ):
                self._rpc[f"env.{name}"] = self.unsupported

        @staticmethod
        def unsupported(*args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("RPent VLA integration does not support this operation")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    args = parser.parse_args(argv)
    try:
        from manimux.integrations.rpent.environment import YamEnvironment
    except ModuleNotFoundError as error:
        parser.error(f"Install RPent and its base dependencies in this environment: {error}")
    from manimux.cli import (
        _create_run_dir,
        _runtime_lock,
        load_config,
        load_local,
        resolve_local_path,
    )
    from manimux.policies.base import action_interval
    from manimux.runtime.edge import EdgeRuntime
    from manimux.runtime.external import ExternalActionSource

    logging.basicConfig(level=logging.INFO)
    station = resolve_local_path(args.experiment, args.local)
    config = load_config(args.experiment, local=station)
    bindings = load_local(station)
    for sensor in config["sensors"]:
        service = sensor.get("service")
        if service in bindings.get("services", {}):
            sensor["options"]["endpoint"] = bindings["services"][service]["request_endpoint"]
    if config["robot"]["group_dims"] != {"left_arm": 7, "right_arm": 7}:
        parser.error("RPent YAM requires left_arm/right_arm groups with seven coordinates each")
    if (
        config["robot"]["control_hz"] != 30
        or abs(action_interval(config["policy"]) - 1 / 30) > 1e-10
    ):
        parser.error("RPent YAM requires 30 Hz control and policy actions")
    source = ExternalActionSource()
    with _runtime_lock(config, args.experiment, mode="rpent"):
        run_dir = _create_run_dir(config, args.experiment, mode="rpent")
        runtime = EdgeRuntime(config, run_dir, external=source)
        environment = YamEnvironment(source, runtime._robot.kinematics, config["rpent"])
        facade = build_facade(environment)

        def run() -> None:
            try:
                runtime.run()
            except BaseException as error:
                logging.exception("RPent execution runtime stopped")
                source.failure = error
            finally:
                source.close()
                facade._shutdown_event.set()

        worker = threading.Thread(target=run, name="manimux-external-runtime")
        worker.start()
        try:
            with source.condition:
                source.condition.wait_for(
                    lambda: source.snapshot is not None or source.failure, timeout=30
                )
            if source.failure or source.snapshot is None:
                raise RuntimeError("ManiMux runtime did not become ready") from source.failure
            environment.observe()  # Validate RGB-D before accepting RPent connections.
            facade.serve(transport="http", host=args.host, port=args.port)
        finally:
            environment.close()
            worker.join()
    return 1 if source.failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
