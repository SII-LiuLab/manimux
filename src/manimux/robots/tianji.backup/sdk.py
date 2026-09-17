"""Load the Tianji vendor SDKs on demand.

The Marvin arm SDK (Apache-2.0, 上海孚晞科技) is vendored under ``vendor/marvin``:
its ctypes bindings plus the Linux x86-64 libraries, which the bindings load
from their own directory, and the ``ccs_m6_40.MvKDCfg`` arm table. ``sdk_root``
points at a different SDK checkout instead (``SDK_PYTHON`` and ``CommonConfig``).

The TacCap gripper SDK (``xense.taccap``) is a native build installed into the
hardware environment. As in CalibWrist's ``real_run.py``, ``xense.taccap`` has
to be imported before the Marvin bindings: the SDK ships an old JPEG library
sharing a SONAME with TacCap's OpenCV dependency.
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
KINE_CONFIG_NAME = "ccs_m6_40.MvKDCfg"


def load_taccap() -> ModuleType:
    try:
        return importlib.import_module("xense.taccap")
    except ImportError as exc:
        raise ImportError(
            "xense.taccap is not importable in this environment. Build and install "
            "TacCap-Gripper into the hardware environment, or set "
            "robot.options.end_effector to none to run without grippers."
        ) from exc


def kine_config(sdk_root: Path | str | None = None) -> Path:
    """The M6 v4.0 kinematics/limits table used by the bindings."""

    if sdk_root is None:
        return VENDOR_MARVIN / KINE_CONFIG_NAME
    return Path(sdk_root).expanduser().resolve() / "CommonConfig" / KINE_CONFIG_NAME


def load_marvin_robot(sdk_root: Path | str | None = None) -> ModuleType:
    """Return the vendor ``fx_robot`` binding (controller link and commands)."""

    return _load_binding("fx_robot", sdk_root)


def load_marvin_kine(sdk_root: Path | str | None = None) -> ModuleType:
    """Return the vendor ``fx_kine`` binding (kinematics)."""

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
