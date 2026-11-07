"""One resident XPolicyLab model, routed by a config list's WebSocket URL paths.

Only catalog-declared recipes are executable. No hardware is owned here. Active
connections lease their model; another model is refused until they disconnect.
The normal XPolicyLab wire protocol, identity checks and RESET fence pass intact.
"""

import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


def route_config(config, source):
    if config["policy"]["worker"] != "xpolicylab_ws":
        raise ValueError("On-demand model switching requires xpolicylab_ws")
    endpoint = config["policy"]["options"]["server"]
    address = urlsplit(endpoint)
    if (
        address.scheme != "ws"
        or not address.hostname
        or address.port is None
        or address.path not in ("", "/")
        or address.query
        or address.fragment
        or address.username
    ):
        raise ValueError(
            "Model catalog endpoint must be ws://host:port without path or credentials"
        )
    recipe = config["policy_server"]
    if recipe.get("protocol", "ws") != "ws":
        raise ValueError("Model catalogs require WebSocket policy recipes")
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()
    path = "/models/" + digest
    config["policy"]["options"]["server"] = urlunsplit(address._replace(path=path))
    config["policy"]["options"]["model_catalog"] = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "endpoint": urlunsplit(address._replace(path="")),
    }
    return path


def resolve_catalog(path, local=None):
    from manimux.cli import read_experiment
    from manimux.collections import _read_collection

    source, raw, _, station = _read_collection(path, local)
    if raw.get("policy_switching") != "on_demand":
        return None
    routes = {}
    endpoint = binding = None
    for entry in raw["experiments"]:
        config = read_experiment(source.parent / entry["config"], local=station)
        route = route_config(config, source)
        current = config["policy"]["options"]["model_catalog"]["endpoint"]
        recipe = config["policy_server"]
        bind = (recipe["host"], int(recipe["port"]))
        if endpoint is not None and (current != endpoint or binding != bind):
            raise ValueError("All model catalog entries must share one policy endpoint/binding")
        endpoint, binding = current, bind
        routes.setdefault(route, {"experiment_id": entry["id"], "recipe": recipe})
    return {
        "source": source,
        "local": station,
        "endpoint": endpoint,
        "binding": binding,
        "routes": routes,
    }


class ModelCatalogServer:
    def __init__(self, catalog, *, startup_timeout_s=120.0):
        self.catalog = catalog
        self.startup_timeout_s = startup_timeout_s
        self.lock = asyncio.Lock()
        self.process = None
        self.route = None
        self.upstream = None
        self.leases = 0
        self.server = None
        self.recipe_directory = None
        self.closing = False

    async def _stop_model(self):
        process, self.process = self.process, None
        self.route = self.upstream = None
        try:
            if process is not None and process.returncode is None:
                # This child is owned by this router, never a discovered service PID.
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            if self.recipe_directory is not None:
                self.recipe_directory.cleanup()
                self.recipe_directory = None

    async def _load_model(self, route):
        from websockets.asyncio.client import connect

        if self.route == route and self.process is not None and self.process.returncode is None:
            return
        await self._stop_model()
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        entry = self.catalog["routes"][route]
        logger.info("Loading model for %s", entry["experiment_id"])
        self.recipe_directory = tempfile.TemporaryDirectory(prefix="manimux-model-")
        recipe_path = Path(self.recipe_directory.name) / "recipe.json"
        recipe_path.write_text(
            json.dumps({**entry["recipe"], "host": "127.0.0.1", "port": port}, default=str)
        )
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "manimux.servers.policy",
            "--resolved-recipe",
            str(recipe_path),
            "--parent-pid",
            str(os.getpid()),
        )
        self.upstream = f"ws://127.0.0.1:{port}"
        deadline = asyncio.get_running_loop().time() + self.startup_timeout_s
        try:
            while self.process.returncode is None:
                if self.closing:
                    raise asyncio.CancelledError
                try:
                    async with connect(
                        self.upstream, max_size=None, compression=None, open_timeout=1
                    ):
                        pass
                except (OSError, TimeoutError):
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("Model startup exceeded catalog timeout") from None
                    await asyncio.sleep(0.1)
                else:
                    self.route = route
                    logger.info("Model ready for %s", entry["experiment_id"])
                    return
            raise RuntimeError(f"Model process exited with code {self.process.returncode}")
        except BaseException:
            await self._stop_model()
            raise

    async def handle(self, connection):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        route = connection.request.path
        if route not in self.catalog["routes"]:
            await connection.close(1008, "Unknown model route; use the same config list")
            return
        leased = False
        try:
            async with self.lock:
                if self.leases and self.route != route:
                    await connection.close(1013, "Model in use; finish the active rollout first")
                    return
                await self._load_model(route)
                self.leases += 1
                leased = True
                upstream = self.upstream
            async with connect(upstream, max_size=None, compression=None) as backend:

                async def forward(source, target):
                    async for message in source:
                        await target.send(message)

                tasks = [
                    asyncio.create_task(forward(connection, backend)),
                    asyncio.create_task(forward(backend, connection)),
                ]
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("Model route failed")
            await connection.close(1011, "Model unavailable; see policy service log")
        finally:
            if leased:
                async with self.lock:
                    self.leases -= 1
            await connection.close()

    async def start(self):
        from websockets.asyncio.server import serve

        self.server = await serve(
            self.handle,
            *self.catalog["binding"],
            max_size=None,
            compression=None,
        )

    async def close(self):
        self.closing = True
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        await self._stop_model()

    async def run(self):
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        try:
            await self.start()
            logger.info("Model catalog listening at %s", self.catalog["endpoint"])
            await stopped.wait()
        finally:
            await self.close()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with suppress(NotImplementedError):
                    loop.remove_signal_handler(sig)
