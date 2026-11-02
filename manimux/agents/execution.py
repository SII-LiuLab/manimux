"""RPent experiment parameters; no planner dependency or hardware construction."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from manimux.config_validation import check_fields


def default_planner_python() -> str:
    """RPent's own environment when installed, otherwise the runtime interpreter."""
    planner = Path(__file__).resolve().parents[2] / "third_party/RPent/.venv/bin/python"
    return str(planner) if planner.exists() else sys.executable


def agent_parameters(options: dict) -> dict:
    check_fields(options, {"config", "python", "debug_max_plan_age_s", "policy_enabled"}, "agent")
    if not isinstance(options.get("config"), str) or not options["config"].strip():
        raise ValueError("agent.config must reference an RPent recipe")
    python = options.get("python", default_planner_python())
    if not isinstance(python, str) or not python.strip():
        raise ValueError("agent.python must name the planner Python executable")
    values = {"config": options["config"], "python": python}
    if "policy_enabled" in options:
        if type(options["policy_enabled"]) is not bool:
            raise ValueError("agent.policy_enabled must be a boolean")
        values["policy_enabled"] = options["policy_enabled"]
    if "debug_max_plan_age_s" in options:
        age = options["debug_max_plan_age_s"]
        if (isinstance(age, bool) or not isinstance(age, (int, float))
                or not math.isfinite(age) or age <= 0):
            raise ValueError("agent.debug_max_plan_age_s must be a finite positive number")
        values["debug_max_plan_age_s"] = float(age)
    return values


def execution_choices(config: dict) -> list[str]:
    # An agent profile makes this an RPent experiment; plain VLA runs use ordinary recipes.
    return ["rpent"] if config.get("agent") else ["vla"]


def execution_condition(condition: str, mode: str) -> str:
    """Keep VLA-only study identity intact and count RPent trials separately."""
    if condition == "RPent" or condition.endswith(" / RPent"):
        return condition
    return f"{condition} / RPent" if mode == "rpent" and condition else (
        "RPent" if mode == "rpent" else condition
    )
