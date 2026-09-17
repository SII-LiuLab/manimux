from __future__ import annotations

import importlib
from importlib import metadata
from typing import TypeVar, cast


class PluginError(ValueError):
    """Raised when a configured ManiMux plugin cannot be resolved."""


PluginT = TypeVar("PluginT")


def load_plugin(
    name: str,
    *,
    group: str,
    builtins: dict[str, PluginT | str],
) -> PluginT:
    """Resolve a built-in, entry-point, or ``module:attribute`` plugin.

    Built-ins keep the checked-in V1 configuration stable. Entry points make
    separately installed policy/robot packages discoverable, while the explicit
    module form is useful during local development without an installation step.
    Built-in values may also be module references, keeping SDK imports lazy
    without a separate forwarding function for every registry entry.
    """

    if name in builtins:
        plugin = builtins[name]
        if not isinstance(plugin, str):
            return plugin
        name = plugin

    if ":" in name:
        module_name, attribute = name.split(":", 1)
        # 保留 Python 的导入异常和原始 traceback，直接定位插件中的错误。
        module = importlib.import_module(module_name)
        return cast(PluginT, getattr(module, attribute))

    matches = [entry for entry in metadata.entry_points(group=group) if entry.name == name]
    if not matches:
        available = sorted(
            {*builtins, *(entry.name for entry in metadata.entry_points(group=group))}
        )
        choices = ", ".join(available) or "none"
        raise PluginError(f"unknown {group} plugin {name!r}; available: {choices}")
    if len(matches) > 1:
        raise PluginError(f"multiple {group} entry points are registered as {name!r}")
    return cast(PluginT, matches[0].load())
