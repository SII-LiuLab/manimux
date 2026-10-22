"""Strict public configuration fields; plugin-owned option mappings stay extensible."""

from collections.abc import Mapping
from difflib import get_close_matches


def check_fields(options: Mapping, allowed, section: str) -> None:
    if not isinstance(options, Mapping):
        raise ValueError(f"{section} must be a mapping")
    unknown = options.keys() - set(allowed)
    if unknown:
        errors = []
        for key in sorted(unknown, key=str):
            matches = get_close_matches(str(key), allowed, n=1, cutoff=0.65)
            hint = f"; did you mean {section}.{matches[0]}?" if matches else ""
            errors.append(f"{section}.{key}{hint}")
        raise ValueError("Unknown configuration field: " + ", ".join(errors))


def merge_parameters(defaults: dict, options: Mapping, section: str, *, extra=()) -> dict:
    check_fields(options, {*defaults, *extra}, section)
    return {**defaults, **options}


EXPERIMENT_FIELDS = {
    "agent",
    "run",
    "robot",
    "policy",
    "policy_server",
    "camera_server",
    "sensors",
    "inference",
    "executor",
    "execution",
    "robogui",
    "recording",
    "evaluation",
    "control_profile",
    "local",
    "experiment",
    "initial_pose",
    "config_sources",
    "collection",
}
