"""Run SmolVLA in a selected Python environment over private inherited pipes."""

from __future__ import annotations

import argparse
import io
import os
import pickle
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from manimux.policies.base import PolicyModel
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.response_log import capture_response
from manimux.types import InferenceRequest


def _reduce_array(
    array: NDArray[Any],
) -> tuple[Callable[..., NDArray[Any]], tuple[object, ...]]:
    # Use the public ndarray constructor across different NumPy installations.
    return np.ndarray, (array.shape, array.dtype.str, bytearray(array.tobytes()))


def _send(connection: Connection, value: object) -> None:
    with io.BytesIO() as buffer:
        pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dispatch_table = {np.ndarray: _reduce_array}
        pickler.dump(value)
        connection.send_bytes(buffer.getvalue())


class IsolatedSmolVLAModel:
    """PolicyModel proxy with no Torch or LeRobot imports."""

    def __init__(self, config: dict[str, Any], python: str | Path) -> None:
        # Resolving an executable symlink would discard its virtualenv identity.
        executable = Path(python).expanduser().absolute()
        self._connection, remote = Pipe()
        owner_reader, self._owner = Pipe(duplex=False)
        self._supervisor: subprocess.Popen[bytes] | None = None
        try:
            with remote, owner_reader:
                channel_fd, owner_fd = remote.fileno(), owner_reader.fileno()
                self._supervisor = subprocess.Popen(
                    [
                        str(executable),
                        "-I",
                        "-m",
                        "manimux.policies.lerobot.process",
                        str(channel_fd),
                        str(owner_fd),
                    ],
                    stdin=subprocess.DEVNULL,
                    pass_fds=(channel_fd, owner_fd),
                    start_new_session=True,
                )
            _send(self._connection, config)
            timeout = config["startup_timeout_s"]
            if not self._connection.poll(timeout):
                raise TimeoutError(f"Isolated SmolVLA did not load within {timeout:.1f}s")
            capabilities = self._receive()
            if not isinstance(capabilities, PolicyCapabilities):
                raise TypeError("Isolated SmolVLA returned invalid capabilities")
            self._capabilities = capabilities
        except BaseException:
            self.close()
            raise

    def _receive(self) -> object:
        while True:
            try:
                kind, value = self._connection.recv()
            except (EOFError, OSError) as exc:
                self._connection.close()
                raise RuntimeError("Isolated SmolVLA disconnected") from exc
            if kind == "ok":
                return value
            if kind == "error":
                raise RuntimeError(value)
            if kind != "smolvla_prediction":
                self._connection.close()
                raise RuntimeError(f"Unexpected isolated SmolVLA reply {kind!r}")
            try:
                capture_response(value, stage=kind)
                self._connection.send("captured")
            except Exception:
                self.close()
                raise

    def capabilities(self) -> PolicyCapabilities:
        return self._capabilities

    def backend_is_alive(self) -> bool:
        return (
            not self._connection.closed
            and self._supervisor is not None
            and self._supervisor.poll() is None
        )

    def _call(self, operation: str, value: object) -> object:
        try:
            _send(self._connection, (operation, value))
        except (EOFError, OSError) as exc:
            self._connection.close()
            raise RuntimeError("Isolated SmolVLA disconnected") from exc
        return self._receive()

    def reset(self, session_id: str) -> None:
        self._call("reset", session_id)

    def infer(self, request: InferenceRequest) -> object:
        return self._call("infer", request)

    def close(self) -> None:
        self._connection.close()
        try:
            if self._supervisor is not None:
                try:
                    self._supervisor.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._owner.close()
                    try:
                        self._supervisor.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(self._supervisor.pid, signal.SIGKILL)
                        self._supervisor.wait()
        finally:
            self._owner.close()


def build_model(config: dict[str, Any]) -> PolicyModel:
    python = config["options"].get("model_python")
    if python is not None:
        return IsolatedSmolVLAModel(config, python)

    from manimux.policies.lerobot.client import SmolVLAPolicyModel

    return SmolVLAPolicyModel(config)


def _supervise(channel_fd: int, owner_fd: int) -> None:
    # The supervisor can reap the model even while a native call holds its GIL.
    with Connection(channel_fd) as channel:
        try:
            model = subprocess.Popen(
                [sys.executable, "-I", "-m", "manimux.policies.lerobot.process", str(channel_fd)],
                stdin=subprocess.DEVNULL,
                pass_fds=(channel_fd,),
            )
        except OSError as exc:
            channel.send(("error", f"Could not start isolated SmolVLA: {exc}"))
            return
    with Connection(owner_fd, readable=True, writable=False) as owner:
        try:
            while model.poll() is None and not owner.poll(0.05):
                pass
        finally:
            if model.poll() is None:
                model.kill()
            model.wait()


def _serve(connection: Connection) -> None:
    def capture_prediction(payload: dict[str, object]) -> None:
        _send(connection, ("smolvla_prediction", payload))
        # The owning worker persists the raw output before validation continues.
        if connection.recv() != "captured":
            raise RuntimeError("Missing acknowledgement for the SmolVLA prediction")

    config = connection.recv()
    try:
        from manimux.policies.lerobot.client import SmolVLAPolicyModel

        logging_enabled = config["options"].get("response_log_dir") is not None
        model = SmolVLAPolicyModel(
            config, capture_prediction=capture_prediction if logging_enabled else None
        )
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    try:
        connection.send(("ok", model.capabilities()))
        while True:
            operation, value = connection.recv()
            try:
                result = None
                if operation == "reset":
                    model.reset(value)
                elif operation == "infer":
                    result = model.infer(value)
                else:
                    raise ValueError(f"Unknown SmolVLA operation {operation!r}")
                _send(connection, ("ok", result))
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel_fd", type=int, nargs="?")
    parser.add_argument("owner_fd", type=int, nargs="?")
    parser.add_argument("--check", action="store_true", help="Check model environment imports")
    args = parser.parse_args()
    if args.check:
        from manimux.policies.lerobot.client import SmolVLAPolicyModel

        print(f"{SmolVLAPolicyModel.__name__} imports OK")
        return
    if args.channel_fd is None:
        parser.error("Start the model using the lerobot_smolvla policy worker")
    if args.owner_fd is not None:
        _supervise(args.channel_fd, args.owner_fd)
        return
    with (
        Connection(args.channel_fd) as connection,
        suppress(EOFError, BrokenPipeError, ConnectionResetError),
    ):
        _serve(connection)


if __name__ == "__main__":
    main()
