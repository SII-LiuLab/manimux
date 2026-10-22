"""GUI execution selection; no planner dependency or hardware construction."""

from __future__ import annotations

import math
import sys

from manimux.config_validation import check_fields

EXECUTION_LABELS = {"vla": "Direct VLA", "rpent": "RPent"}


def agent_parameters(options: dict) -> dict:
    check_fields(options, {"config", "python", "debug_max_plan_age_s", "policy_enabled"}, "agent")
    if not isinstance(options.get("config"), str) or not options["config"].strip():
        raise ValueError("agent.config must reference an RPent recipe")
    python = options.get("python", sys.executable)
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
    return ["vla", "rpent"] if config.get("agent") else ["vla"]


def execution_condition(condition: str, mode: str) -> str:
    """Keep direct VLA study identity intact and count RPent trials separately."""
    if condition == "RPent" or condition.endswith(" / RPent"):
        return condition
    return f"{condition} / RPent" if mode == "rpent" and condition else (
        "RPent" if mode == "rpent" else condition
    )
