"""Read-only experiment identity snapshots without importing model or SDK runtimes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_SOURCE_SUFFIXES = {".py", ".pyi", ".yaml", ".yml", ".toml", ".sh", ".xml", ".urdf", ".xacro"}
_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "checkpoints",
    "credentials",
    "data",
    "datasets",
    "envs",
    "node_modules",
    "outputs",
    "secrets",
    "tests",
    "training",
    "videos",
    "weights",
}


def _source_path(path: str) -> bool:
    value = Path(path)
    parts = set(value.parts)
    if parts & _EXCLUDED_PARTS or "configs/local" in value.as_posix():
        return False
    if re.search(r"(^|[_.-])(token|secret|credential|password)([_.-]|$)", value.name, re.I):
        return False
    return value.suffix in _SOURCE_SUFFIXES or value.name in {"uv.lock", ".gitmodules"}


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="surrogateescape")
    except (OSError, subprocess.SubprocessError):
        return None


def _file_identity(path: Path) -> dict:
    """Hash bytes only; do not copy source, follow symlinks or read large artifacts."""
    if path.is_symlink():
        return {"sha256": None, "reason": "symlink_not_followed"}
    try:
        before = path.stat()
        if not path.is_file():
            return {"sha256": None, "reason": "not_a_regular_file"}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        return {
            "sha256": digest.hexdigest() if stable else None,
            "size_bytes": after.st_size,
            "mode": after.st_mode & 0o777,
            "stable_read": stable,
            **({} if stable else {"reason": "changed_during_read"}),
        }
    except OSError as exc:
        return {"sha256": None, "reason": type(exc).__name__}


def _fingerprint(entries: dict) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def _repository_source(root: Path, paths: list[str]) -> dict:
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.strip()).resolve() != root.resolve():
        return {"available": False, "reason": "source_checkout_unavailable"}
    head = _git(root, "rev-parse", "HEAD")
    tracked = _git(root, "ls-files", "-z", "--cached", "--", *paths)
    untracked = _git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", *paths)
    if head is None or tracked is None or untracked is None:
        return {"available": False, "reason": "git_source_identity_unavailable"}
    tracked_paths = set(tracked.split("\0")) - {""}
    untracked_paths = set(untracked.split("\0")) - {""}
    names = sorted(name for name in tracked_paths | untracked_paths if _source_path(name))
    files = {
        name: {"tracked": name in tracked_paths, **_file_identity(root / name)} for name in names
    }
    changed = _git(root, "diff", "--name-only", "-z", "HEAD", "--", *paths)
    changed_names = (
        None
        if changed is None
        else sorted(
            name for name in set(changed.split("\0")) | untracked_paths if _source_path(name)
        )
    )
    result = {
        "available": True,
        "head": head.strip(),
        "scope": paths,
        "dirty_source": None if changed_names is None else bool(changed_names),
        "changed_source_paths": changed_names,
        "source_sha256": _fingerprint(files),
        "files": files,
        "complete": all(entry.get("sha256") is not None for entry in files.values()),
        "submodules": {},
    }
    indexed = _git(root, "ls-files", "--stage", "-z", "--", *paths)
    if indexed is None:
        result["submodules"] = None
        result["complete"] = False
    else:
        for entry in indexed.split("\0"):
            if not entry.startswith("160000 "):
                continue
            declaration, name = entry.split("\t", 1)
            child = _repository_source(root / name, ["."])
            child["index_commit"] = declaration.split()[1]
            result["submodules"][name] = child
            result["complete"] &= bool(child.get("complete", False))
        result["source_sha256"] = _fingerprint(
            {
                "files": files,
                "submodules": result["submodules"],
            }
        )
    return result


def capture_source_provenance(repository_root: Path, *, policy_name: str | None = None) -> dict:
    """Identify selected runtime/model source, including untracked source files.

    Data, credentials, local station files and file contents are never included.
    This is a read-time fingerprint, not an atomic Git snapshot or a source archive.
    """
    roots = {
        "manimux": _repository_source(
            repository_root, ["manimux", "scripts", "pyproject.toml", "uv.lock", ".gitmodules"]
        ),
    }
    if policy_name is not None and re.fullmatch(r"[A-Za-z0-9_]+", policy_name):
        roots["xpolicylab"] = _repository_source(
            repository_root / "XPolicyLab",
            [
                "client_server",
                "utils",
                "model_template.py",
                "setup_policy_server.py",
                "XPolicyLab.py",
                "__init__.py",
                "pyproject.toml",
                ".gitmodules",
                f"policy/{policy_name}",
            ],
        )
    return {
        "schema": "source-provenance-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "repositories": roots,
        "coverage": "selected_runtime_and_model_source_only",
        "excluded": "data, weights, local station, credentials, tests, docs, environments",
    }


def installed_package_provenance(name: str) -> dict:
    """Read installed distribution identity and effective on-disk SDK source hashes."""
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"name": name, "available": False, "reason": "distribution_not_found"}
    result = {"name": name, "available": True, "version": distribution.version}
    direct = distribution.read_text("direct_url.json")
    if direct:
        try:
            source = json.loads(direct)
            url = urlsplit(source.get("url", ""))
            host = url.hostname or ""
            if url.port is not None:
                host += f":{url.port}"
            vcs = source.get("vcs_info", {})
            result["source"] = {
                "url": urlunsplit((url.scheme, host, url.path, "", "")),
                "vcs": vcs.get("vcs"),
                "commit_id": vcs.get("commit_id"),
            }
        except (TypeError, ValueError):
            result["source"] = {"available": False, "reason": "invalid_direct_url_metadata"}
    else:
        result["source"] = {"available": False, "reason": "direct_url_metadata_missing"}
    paths = distribution.files
    if paths is None:
        result["source_files"] = None
        result["source_files_reason"] = "distribution_file_list_missing"
    else:
        files = {
            str(path): _file_identity(Path(distribution.locate_file(path)))
            for path in paths
            if str(path).startswith(f"{name}/") and _source_path(str(path))
        }
        result["source_files"] = files
        result["source_sha256"] = _fingerprint(files)
        result["source_complete"] = bool(files) and all(
            entry.get("sha256") is not None for entry in files.values()
        )
    return result


def loaded_module_provenance(name: str) -> dict:
    """Identify a module already loaded by the connected controller; never import it."""
    module = sys.modules.get(name)
    source = None if module is None else getattr(module, "__file__", None)
    if source is None:
        return {"module": name, "available": False, "reason": "module_source_unavailable"}
    path = Path(source)
    if path.suffix not in {".py", ".pyi"}:
        return {"module": name, "available": False, "reason": "module_is_not_python_source"}
    return {"module": name, "path": str(path), **_file_identity(path)}


def deployment_artifact_provenance(recipe: Mapping) -> dict:
    """Keep deployment declarations/stat identity; do not hash checkpoint payloads."""
    result = {}
    for key in ("model_path", "norm_stats_path"):
        value = recipe.get(key)
        if not value:
            result[key] = {"available": False, "reason": "not_declared"}
            continue
        path = Path(value).expanduser()
        entry = {"path": str(path), "content_sha256": None, "hash_reason": "not_hashed"}
        try:
            stat = path.stat()
            entry.update(
                available=True,
                is_directory=path.is_dir(),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        except OSError as exc:
            entry.update(available=False, reason=type(exc).__name__)
        result[key] = entry
    result["declared_identity"] = {
        key: recipe[key]
        for key in (
            "checkpoint_source",
            "checkpoint_variant",
            "norm_stats_source",
            "train_config_name",
        )
        if key in recipe
    }
    return result


def state_evidence(state, *, targets: Mapping | None = None) -> dict:
    """Freeze measured coordinates and target errors without imposing arrival criteria."""
    groups = {name: value.tolist() for name, value in state.groups.items()}
    result = {
        "monotonic_ns": int(state.monotonic_ns),
        "sequence": int(state.sequence),
        "timestamp_source": "robot_state_host_receipt",
        "groups": groups,
        "target_groups": None,
        "target_error": None,
    }
    if targets is not None:
        result["target_groups"] = {
            name: [float(coordinate) for coordinate in value] for name, value in targets.items()
        }
        if set(targets) == set(groups) and all(
            len(targets[name]) == len(groups[name]) for name in groups
        ):
            result["target_error"] = {
                name: [
                    float(actual - target)
                    for actual, target in zip(groups[name], targets[name], strict=True)
                ]
                for name in groups
            }
        else:
            result["target_error_reason"] = "target_shape_mismatch"
    else:
        result["target_error_reason"] = "target_not_supplied"
    return result
