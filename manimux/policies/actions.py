"""Backend-independent action payloads consumed by embodiment adapters.

A payload contains format (joint or pose), actions keyed by robot group, and
optional semantics/sampler metadata. Joint rows retain the declared arm/tool
coordinate order. Pose rows are xyz + quaternion wxyz + tool coordinates, in
metres and the explicitly declared action frame. No normalization or IK occurs
here; relative/absolute interpretation remains with the selected adapter.
"""

from collections.abc import Mapping

import numpy as np


def action_groups(raw, dimensions, *, format):
    if not isinstance(raw, Mapping) or raw.get("format") != format:
        raise ValueError(f"adapter requires a canonical {format} action payload")
    actions = raw.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(dimensions):
        raise ValueError("action groups must match the configured robot groups")
    groups = {}
    for name, width in dimensions.items():
        values = np.asarray(actions[name], dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != width
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"invalid action matrix for group {name!r}, expected (steps, {width})")
        groups[name] = np.ascontiguousarray(values)
    if len({len(values) for values in groups.values()}) != 1:
        raise ValueError("action groups must have the same horizon")
    return groups
