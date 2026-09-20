"""Inference-time Physical Intelligence RTC guidance for MolmoAct2.

MolmoAct2 integrates a flow from noise at ``t=0`` to actions at ``t=1``.
The RTC client supplies an old-chunk continuation in robot units plus the
paper's soft prefix weights.  :class:`MolmoActRTC` normalizes that target with
the checkpoint statistics and installs a small wrapper around the action
expert's Euler loop.  No weights or cached Hugging Face sources are modified.
"""

from __future__ import annotations

import contextlib
import math
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class _RTCCondition:
    target: np.ndarray
    weights: np.ndarray
    beta: float


def _guidance_scale(flow_time: float, beta: float) -> float:
    """PiGDM guidance coefficient for a noise-at-0, actions-at-1 flow."""
    remaining = 1.0 - flow_time
    if flow_time <= 0.0:
        return beta
    if remaining <= 0.0:
        return 0.0
    r_squared = remaining * remaining / (flow_time * flow_time + remaining * remaining)
    return min(beta, remaining / (flow_time * r_squared))


def guided_action_flow_loop(
    model: Any,
    inputs: Any,
    steps: int,
    condition: _RTCCondition,
) -> torch.Tensor:
    """Run MolmoAct2's Euler sampler with PI RTC guidance at every step."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    action_expert = model._require_action_expert()
    dt = 1.0 / steps
    trajectory = inputs.trajectory
    target = torch.as_tensor(
        condition.target, device=trajectory.device, dtype=trajectory.dtype
    ).unsqueeze(0)
    weights = torch.as_tensor(
        condition.weights, device=trajectory.device, dtype=trajectory.dtype
    ).view(1, -1, 1)
    if target.shape[0] == 1 and trajectory.shape[0] != 1:
        target = target.expand(trajectory.shape[0], -1, -1)
    if target.shape != trajectory.shape:
        raise ValueError(
            f"normalized RTC target must match trajectory {tuple(trajectory.shape)}, "
            f"got {tuple(target.shape)}"
        )

    action_dim_is_pad = inputs.action_dim_is_pad
    mask_enabled = model.config.mask_action_dim_padding
    for idx in range(steps):
        flow_time = idx / steps
        # generate_actions_from_inputs is no-grad by design. RTC needs only the
        # VJP with respect to the action trajectory, never parameter gradients.
        with torch.enable_grad():
            x_t = trajectory.detach().requires_grad_(True)
            velocity = action_expert.forward_with_context(
                x_t,
                inputs.modulations[idx].conditioning,
                context=inputs.context,
                modulation=inputs.modulations[idx],
            )
            velocity = model._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=mask_enabled,
            )
            action_estimate = x_t + (1.0 - flow_time) * velocity
            weighted_error = (target - action_estimate) * weights
            correction = torch.autograd.grad(
                outputs=action_estimate,
                inputs=x_t,
                grad_outputs=weighted_error.detach(),
                create_graph=False,
                retain_graph=False,
            )[0]

        guided_velocity = velocity + _guidance_scale(flow_time, condition.beta) * correction
        trajectory = (x_t + dt * guided_velocity).detach()
        trajectory = model._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=mask_enabled,
        )
    return trajectory


class MolmoActRTC:
    """Permanent flow-loop wrapper with a per-inference RTC condition."""

    def __init__(self, model: Any, norm_tag: str) -> None:
        self.model = model
        self.flow_model = model.model
        self.norm_tag = str(norm_tag)
        self._active: _RTCCondition | None = None
        original = self.flow_model._run_action_flow_loop

        def wrapped(flow_model: Any, inputs: Any, steps: int) -> torch.Tensor:
            condition = self._active
            if condition is None:
                return original(inputs, steps)
            return guided_action_flow_loop(flow_model, inputs, steps, condition)

        self.flow_model._run_action_flow_loop = types.MethodType(wrapped, self.flow_model)

    @contextlib.contextmanager
    def condition(
        self,
        action_condition: Any | None,
        condition_weights: Any | None,
        beta: float,
    ) -> Iterator[bool]:
        """Activate one normalized RTC condition; yield whether RTC is active."""
        if action_condition is None and condition_weights is None:
            yield False
            return
        if action_condition is None or condition_weights is None:
            raise ValueError("action_condition and action_condition_weights must be sent together")
        if self._active is not None:
            raise RuntimeError("nested/concurrent MolmoAct RTC inference is unsupported")
        beta = float(beta)
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"rtc_beta must be finite and positive, got {beta}")

        stats = self.model._get_robot_stats()
        tag = stats.validate_tag(self.norm_tag)
        action_dim = int(stats.get_action_dim(tag) or self.flow_model.config.max_action_dim)
        horizon = int(stats.get_action_horizon(tag) or self.flow_model._resolve_action_horizon())
        raw_target = np.asarray(action_condition, dtype=np.float32)
        if raw_target.shape != (horizon, action_dim):
            raise ValueError(
                f"action_condition must have shape {(horizon, action_dim)}, got {raw_target.shape}"
            )
        # msgpack_numpy may decode a read-only view; copy before torch wraps it.
        weights = np.array(condition_weights, dtype=np.float32, copy=True)
        if weights.shape != (horizon,):
            raise ValueError(
                f"action_condition_weights must have shape {(horizon,)}, got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("action_condition_weights must be finite and in [0, 1]")

        action_normalizer = stats.action_normalizers.get(tag)
        normalized = np.array(
            raw_target if action_normalizer is None else action_normalizer.normalize(raw_target),
            dtype=np.float32,
            copy=True,
        )
        max_action_dim = int(self.flow_model.config.max_action_dim)
        padded = np.zeros((horizon, max_action_dim), dtype=np.float32)
        padded[:, :action_dim] = normalized
        self._active = _RTCCondition(padded, weights, beta)
        try:
            yield True
        finally:
            self._active = None
