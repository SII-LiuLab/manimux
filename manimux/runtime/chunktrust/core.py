"""Transactional composition around the frozen PI AHS numerical methods.

History belongs to the runtime ledger and is an explicit input on every call.
This adapter never calls the upstream constructor/reset or mutates its classes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from manimux.runtime.chunktrust._pi_ahs import Selector as _FrozenMethods
from manimux.runtime.chunktrust.config import ChunkTrustOptions, _integer
from manimux.runtime.chunktrust.provenance import CORE_ID, core_provenance, verify_frozen_core

_EvidenceKernel = Callable[
    [object, NDArray[np.float32], int, NDArray[np.float32] | None], dict[str, Any]
]
_SelectionKernel = Callable[[object, dict[str, Any], int], tuple[int, dict[str, Any]]]


def _json_value(value: Any) -> Any:
    """JSON diagnostics use null for undefined inter scores, never fabricated zeros."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ValueError(f"ChunkTrust context/diagnostics cannot serialize {type(value).__name__}")


class _CallFrame:
    """An explicit, short-lived host for unchanged upstream numerical functions."""

    def __init__(
        self,
        options: ChunkTrustOptions,
        horizon: int,
        posterior: dict[int, list[float]],
        rng: np.random.Generator,
        history: NDArray[np.float32],
    ) -> None:
        self.options = options
        self.fft_horizon = horizon
        self._horizon_ts_state = posterior
        self._horizon_ts_rng = rng
        self._executed_action_history = history

    # These fixed read-only properties are the entire parameter-to-kernel mapping.
    horizon_candidates = property(lambda self: self.options.candidates)
    horizon_exec_mode = property(lambda self: self.options.exec_mode)
    horizon_expected_temp = property(lambda self: self.options.temperature)
    horizon_ts_epsilon = property(lambda self: self.options.epsilon)
    horizon_ts_update_mode = property(lambda self: self.options.update_mode)
    horizon_ts_kernel_bandwidth = property(lambda self: self.options.kernel_bandwidth)
    horizon_ts_forget_rho = property(lambda self: self.options.forget_rho)
    horizon_ts_update_eta = property(lambda self: self.options.update_eta)
    horizon_intra_cut_t = property(lambda self: self.options.intra_cut_t)
    horizon_intra_z_mode = property(lambda self: self.options.intra_z_mode)
    horizon_intra_tau_window = property(lambda self: self.options.intra_tau_window)
    horizon_intra_alpha = property(lambda self: self.options.intra_alpha)
    horizon_inter_use_speed_uniformity = property(
        lambda self: self.options.inter_use_speed_uniformity
    )
    horizon_mix_mode = property(lambda self: self.options.mix_mode)
    horizon_inter_window = property(lambda self: self.options.history_window)
    horizon_inter_weight = property(lambda self: self.options.inter_weight)
    horizon_inter_beta = property(lambda self: self.options.inter_beta)
    horizon_inter_fallback = property(lambda self: self.options.inter_fallback)

    _parse_horizon_z_mode = staticmethod(_FrozenMethods._parse_horizon_z_mode)
    _uses_horizon_posterior = _FrozenMethods._uses_horizon_posterior
    _normalize_candidate_metric = staticmethod(_FrozenMethods._normalize_candidate_metric)
    _normalize_score_distribution = staticmethod(_FrozenMethods._normalize_score_distribution)
    _ehh_curves_by_horizon_batched = _FrozenMethods._ehh_curves_by_horizon_batched
    _z_trend_abs_slope = staticmethod(_FrozenMethods._z_trend_abs_slope)
    _compute_z_from_ehh = _FrozenMethods._compute_z_from_ehh
    _speed_uniformity_proxy_by_horizon = _FrozenMethods._speed_uniformity_proxy_by_horizon
    _gaussian_kernel_weights = staticmethod(_FrozenMethods._gaussian_kernel_weights)
    _apply_horizon_ts_update = _FrozenMethods._apply_horizon_ts_update
    # The upstream self annotation names Selector; this explicit call frame
    # supplies the same read-only parameter and scratch-state protocol by composition.
    _compute_horizon_info = cast(_EvidenceKernel, _FrozenMethods._compute_horizon_info)
    _select_exec_k_horizon_from_info = cast(
        _SelectionKernel, _FrozenMethods._select_exec_k_horizon_from_info
    )


@dataclass(frozen=True, slots=True)
class PreparedDecision:
    request_key: str | int
    execution_steps: int
    info: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]
    revision: int
    generation: int
    context: dict[str, Any]

    @property
    def k(self) -> int:
        return self.execution_steps

    def to_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _json_value(
                {
                    "request_key": self.request_key,
                    "execution_steps": self.execution_steps,
                    "info": self.info,
                    "before": self.before,
                    "after": self.after,
                    "revision": self.revision,
                    "generation": self.generation,
                    "context": self.context,
                }
            ),
        )


class ChunkTrustAdapter:
    """One episode-local selector, with speculative prepare and explicit acceptance."""

    def __init__(
        self,
        options: ChunkTrustOptions | Mapping[str, object],
        *,
        horizon: int = 50,
        action_dim: int = 14,
    ) -> None:
        verify_frozen_core()
        self._options = (
            options
            if isinstance(options, ChunkTrustOptions)
            else ChunkTrustOptions.from_mapping(options)
        )
        self._horizon = _integer("profile.horizon", horizon, 1)
        self._action_dim = _integer("profile.action_dim", action_dim, 1)
        if max(self.options.candidates) > self.horizon:
            raise ValueError("chunktrust.candidates exceed the profile prediction horizon")
        self._generation = -1
        self._pending: dict[str | int, tuple[PreparedDecision, dict[str, Any]]] = {}
        self._committed: set[str | int] = set()
        self.reset()

    @property
    def options(self) -> ChunkTrustOptions:
        return self._options

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def generation(self) -> int:
        return self._generation

    def _inputs(
        self,
        velocity: ArrayLike,
        actions: ArrayLike,
        history: ArrayLike,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        velocity, actions = np.asarray(velocity), np.asarray(actions)
        if velocity.ndim != 3 or actions.ndim != 2:
            raise ValueError("ChunkTrust requires velocity[T,H,D] and actions[H,D]")
        if velocity.shape[0] < 1 or velocity.shape[1:] != actions.shape:
            raise ValueError(
                f"ChunkTrust velocity/actions do not align: {velocity.shape}/{actions.shape}"
            )
        if actions.shape[1] != self.action_dim:
            raise ValueError(
                f"ChunkTrust action dimension: expected {self.action_dim}, got {actions.shape[1]}"
            )
        if not max(self.options.candidates) <= len(actions) <= self.horizon:
            raise ValueError(
                "ChunkTrust available suffix cannot support the fixed candidate basis/horizon"
            )
        history = np.asarray(history)
        if history.ndim != 2 or history.shape[1] != self.action_dim:
            raise ValueError(f"ChunkTrust history must have shape [N,{self.action_dim}]")
        for name, value in (("velocity", velocity), ("actions", actions), ("history", history)):
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"ChunkTrust {name} must contain finite real values")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = tuple(
                np.asarray(value, dtype=np.float32) for value in (velocity, actions, history)
            )
        if any(not np.isfinite(value).all() for value in converted):
            raise ValueError("ChunkTrust inputs overflow the reference float32 coordinates")
        return converted[0], converted[1], converted[2][-self.options.history_window :].copy()

    def _frame(self, history: NDArray[np.float32]) -> _CallFrame:
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(self._rng.bit_generator.state)
        return _CallFrame(self.options, self.horizon, copy.deepcopy(self._posterior), rng, history)

    def evaluate(
        self,
        velocity: ArrayLike,
        actions: ArrayLike,
        history: ArrayLike,
    ) -> dict[str, Any]:
        velocity, actions, history = self._inputs(velocity, actions, history)
        frame = self._frame(history)
        info: dict[str, Any] = frame._compute_horizon_info(velocity, self.horizon, actions)
        if not np.isfinite(info["q_mix"]).all():
            raise ValueError("ChunkTrust produced nonfinite candidate evidence")
        info.update(
            fft_horizon=self.horizon,
            available_horizon=len(actions),
            history_rows=len(history),
            parameter_status=self.options.parameter_status(),
        )
        return info

    def prepare(
        self,
        velocity: ArrayLike,
        actions: ArrayLike,
        history: ArrayLike,
        *,
        request_key: str | int,
        context: Mapping[str, object] | None = None,
    ) -> PreparedDecision:
        self._check_key(request_key)
        if request_key in self._pending or request_key in self._committed:
            raise ValueError(f"ChunkTrust duplicate request_key {request_key!r}")
        velocity, actions, history = self._inputs(velocity, actions, history)
        before = self.snapshot()
        frame = self._frame(history)
        info: dict[str, Any] = frame._compute_horizon_info(velocity, self.horizon, actions)
        if not np.isfinite(info["q_mix"]).all():
            raise ValueError("ChunkTrust produced nonfinite candidate evidence")
        k, info = frame._select_exec_k_horizon_from_info(info, len(actions))
        info.update(
            fft_horizon=self.horizon,
            available_horizon=len(actions),
            history_rows=len(history),
            parameter_status=self.options.parameter_status(),
        )
        after = self._snapshot(
            frame._horizon_ts_state,
            frame._horizon_ts_rng,
            self._revision + 1,
            self._committed | {request_key},
        )
        decision = PreparedDecision(
            request_key,
            k,
            info,
            before,
            copy.deepcopy(after),
            self._revision,
            self._generation,
            _json_value(dict(context or {})),
        )
        # The caller cannot change the transaction by editing its returned dictionaries.
        self._pending[request_key] = (decision, after)
        return decision

    @staticmethod
    def _check_key(key: object) -> None:
        if type(key) not in (str, int):
            raise ValueError("ChunkTrust request_key must be a string or integer")

    def commit(self, prepared: PreparedDecision | str | int) -> dict[str, Any]:
        key = prepared.request_key if isinstance(prepared, PreparedDecision) else prepared
        self._check_key(key)
        if key not in self._pending:
            raise ValueError(
                f"ChunkTrust request {key!r} is absent, discarded, or already committed"
            )
        expected, after = self._pending[key]
        if isinstance(prepared, PreparedDecision) and prepared is not expected:
            raise ValueError("ChunkTrust decision does not belong to this transaction")
        if expected.generation != self._generation or expected.revision != self._revision:
            raise ValueError("ChunkTrust decision is stale after a reset or accepted decision")
        self.restore(after)
        return self.snapshot()

    def discard(self, request_key: str | int) -> None:
        self._check_key(request_key)
        self._pending.pop(request_key, None)

    def reset(self, *, seed: int | None = None) -> None:
        self._effective_seed = (
            self.options.seed if seed is None else _integer("reset.seed", seed, 0)
        )
        self._posterior = {k: [1.0, 1.0] for k in self.options.candidates}
        self._rng = np.random.default_rng(self._effective_seed)
        self._generation += 1
        self._revision = 0
        self._pending.clear()
        self._committed.clear()

    def _snapshot(
        self,
        posterior: dict[int, list[float]],
        rng: np.random.Generator,
        revision: int,
        committed: set[str | int],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "core_id": CORE_ID,
            "source": core_provenance(),
            "config": self.options.to_dict(),
            "profile": {"horizon": self.horizon, "action_dim": self.action_dim},
            "posterior": {str(k): list(posterior[k]) for k in self.options.candidates},
            "rng": copy.deepcopy(rng.bit_generator.state),
            "numpy_version": np.__version__,
            "effective_seed": self._effective_seed,
            "generation": self._generation,
            "revision": revision,
            "committed_request_keys": sorted(
                committed, key=lambda key: (type(key).__name__, str(key))
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot(self._posterior, self._rng, self._revision, self._committed)

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        state = copy.deepcopy(dict(snapshot))
        if state.get("schema_version") != 1 or state.get("core_id") != CORE_ID:
            raise ValueError("ChunkTrust snapshot schema/core differs")
        if (
            state.get("source") != core_provenance()
            or state.get("config") != self.options.to_dict()
        ):
            raise ValueError("ChunkTrust snapshot source/config differs")
        if state.get("profile") != {"horizon": self.horizon, "action_dim": self.action_dim}:
            raise ValueError("ChunkTrust snapshot profile differs")
        if state.get("numpy_version") != np.__version__:
            raise ValueError("ChunkTrust exact RNG replay requires the recorded NumPy version")
        if set(state["posterior"]) != {str(k) for k in self.options.candidates}:
            raise ValueError("ChunkTrust snapshot posterior basis differs")
        posterior = {int(k): list(value) for k, value in state["posterior"].items()}
        if set(posterior) != set(self.options.candidates) or any(
            len(pair) != 2
            or any(isinstance(value, bool) for value in pair)
            or np.asarray(pair).dtype.kind not in "fiu"
            or not np.isfinite(pair).all()
            or min(pair) <= 0
            for pair in posterior.values()
        ):
            raise ValueError("ChunkTrust snapshot posterior is invalid")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        generation = _integer("snapshot.generation", state["generation"], 0)
        revision = _integer("snapshot.revision", state["revision"], 0)
        effective_seed = _integer("snapshot.effective_seed", state["effective_seed"], 0)
        committed = state["committed_request_keys"]
        for key in committed:
            self._check_key(key)
        if len(set(committed)) != len(committed):
            raise ValueError("ChunkTrust snapshot repeats a committed request key")
        self._posterior, self._rng = posterior, rng
        self._generation, self._revision, self._effective_seed = (
            generation,
            revision,
            effective_seed,
        )
        self._committed = set(committed)
        self._pending.clear()

    def clone(self) -> ChunkTrustAdapter:
        clone = ChunkTrustAdapter(self.options, horizon=self.horizon, action_dim=self.action_dim)
        clone.restore(self.snapshot())
        return clone
