"""Portable recipe paths. No environment expansion, model imports or device I/O."""

import re
from pathlib import Path

_REFERENCE = re.compile(r"\$\{(repo|paths\.[A-Za-z_][A-Za-z_0-9]*)\}")


def resolve_recipe_paths(value, paths: dict):
    """Expand explicit repository/station roots in nested recipe values."""
    if isinstance(value, dict):
        return {key: resolve_recipe_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_recipe_paths(item, paths) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        key = match[1]
        if key == "repo":
            return str(Path(__file__).resolve().parents[2])
        name = key.removeprefix("paths.")
        if name not in paths:
            raise ValueError(f"Recipe requires station paths.{name}; set it in --local")
        return str(paths[name])

    return _REFERENCE.sub(replace, value)


def read_policy_recipe(path, *, local=None):
    """Read standalone recipes through the same root expansion as collections."""
    from manimux.cli import load_local, read_yaml, resolve_local_path

    recipe = read_yaml(path)
    if not isinstance(recipe, dict):
        raise ValueError(f"Policy recipe must be a mapping: {path}")
    needs_station = "${paths." in str(recipe)
    paths = load_local(resolve_local_path(local=local))["paths"] if needs_station else {}
    return resolve_recipe_paths(recipe, paths)
