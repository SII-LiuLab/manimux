"""Flash ALOHA-style overlap weights, independent of the command executor."""

import math


def sigmoid_smooth_parameters(**options) -> dict:
    unknown = set(options) - {"window_policy_steps", "sigma_policy_steps"}
    if unknown:
        raise ValueError(f"unknown sigmoid_smooth fields: {sorted(unknown)}")
    values = {"window_policy_steps": 4, "sigma_policy_steps": 0.5, **options}
    window = values["window_policy_steps"]
    sigma = values["sigma_policy_steps"]
    if type(window) is not int or window < 1:
        raise ValueError("sigmoid_smooth.window_policy_steps must be a positive integer")
    if (
        isinstance(sigma, bool) or not isinstance(sigma, (int, float))
        or not math.isfinite(sigma) or sigma <= 0
    ):
        raise ValueError("sigmoid_smooth.sigma_policy_steps must be positive and finite")
    return values


def sigmoid_weight(elapsed_ns: int, window_ns: int, sigma_ns: float) -> float:
    """Paper sigmoid inside the overlap; outside it the new plan has full weight.

    The finite-window sigmoid is deliberately not endpoint-normalized: this is
    not a C0/C1 continuity guarantee. The scale stays fixed if overlap is shorter.
    """
    if elapsed_ns >= window_ns:
        return 1.0
    # tanh is equivalent to logistic, without exponential overflow for small sigma.
    return 0.5 * (1.0 + math.tanh((elapsed_ns - window_ns / 2) / (2 * sigma_ns)))
