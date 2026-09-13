"""Load the Tianji vendor SDKs on demand.

The Marvin arm SDK (Apache-2.0, 上海孚晞科技) is vendored under ``vendor/marvin``:
its ctypes bindings plus the Linux x86-64 libraries, which the bindings load
from their own directory. ``sdk_root`` points at a different SDK checkout
instead (its ``SDK_PYTHON`` directory is used).

The TacCap gripper SDK (``xense.taccap``) is a native build installed into the
hardware environment. When both are used in one process, ``xense.taccap`` has to
be imported first; the arm SDK loaded first breaks the TacCap native module's
shared-library resolution on this stack.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

VENDOR_MARVIN = Path(__file__).resolve().parent / "vendor" / "marvin"
KINE_CONFIG = VENDOR_MARVIN / "ccs_m6_40.MvKDCfg"


def load_taccap() -> ModuleType:
    try:
        return importlib.import_module("xense.taccap")
    except ImportError as exc:
        raise ImportError(
            "xense.taccap is not importable in this environment. Build and install "
            "TacCap-Gripper into the hardware environment, or set "
            "robot.options.end_effector to none to run without grippers."
        ) from exc


def load_marvin_robot(sdk_root: Path | str | None = None) -> ModuleType:
    """Return the vendor ``fx_robot`` binding (controller link and commands)."""

    return _load_binding("fx_robot", sdk_root)


def load_marvin_kine(sdk_root: Path | str | None = None) -> ModuleType:
    """Return the vendor ``fx_kine`` binding (closed-form kinematics)."""

    return _load_binding("fx_kine", sdk_root)


def _load_binding(name: str, sdk_root: Path | str | None) -> ModuleType:
    directory = (
        VENDOR_MARVIN if sdk_root is None else Path(sdk_root).expanduser() / "SDK_PYTHON"
    ).resolve()
    path = directory / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Marvin SDK binding not found: {path}")
    digest = hashlib.sha1(str(directory).encode()).hexdigest()[:10]
    module_name = f"_manimux_marvin_{name}_{digest}"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Marvin SDK binding from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        # The bindings print banners on import and construction.
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
