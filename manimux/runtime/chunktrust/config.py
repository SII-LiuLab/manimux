"""The single source of PI-family ChunkTrust AHS runtime parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from typing import Any


def _integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"chunktrust.{name} must be an integer >= {minimum}; got {value!r}")
    return int(value)


def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"chunktrust.{name} must be a finite real number; got {value!r}")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"chunktrust.{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"chunktrust.{name} must be finite; got {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class ChunkTrustOptions:
    """Complete immutable algorithm configuration; dimensions belong to the profile."""

    candidates: tuple[int, ...] = (10, 20, 30, 40, 50)
    exec_mode: str = "expected_round"
    temperature: float = 0.8
    epsilon: float = 0.05
    seed: int = 0
    update_mode: str = "kernel_forget"
    kernel_bandwidth: float = 10.0
    forget_rho: float = 0.99
    update_eta: float = 1.0
    intra_cut_t: float = 0.25
    intra_z_mode: str = "window_rms"
    intra_tau_window: int = 3
    intra_alpha: float = 4.0
    inter_use_speed_uniformity: bool = True
    mix_mode: str = "both"
    history_window: int = 60
    inter_weight: float = 0.5
    inter_beta: float = 4.0
    inter_fallback: str = "ehh_only"

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, list | tuple) or not self.candidates:
            raise ValueError("chunktrust.candidates must be a nonempty list of positive integers")
        candidates = tuple(_integer("candidates", value, 1) for value in self.candidates)
        if list(candidates) != sorted(set(candidates)):
            raise ValueError("chunktrust.candidates must be unique and increasing")
        object.__setattr__(self, "candidates", candidates)
        for name, minimum in (("seed", 0), ("history_window", 4), ("intra_tau_window", 1)):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum))
        ranges = {
            "temperature": (0.0, None, True),
            "epsilon": (0.0, 1.0, False),
            "kernel_bandwidth": (0.0, None, True),
            "forget_rho": (0.0, 1.0, True),
            "update_eta": (0.0, None, False),
            "intra_cut_t": (0.0, 1.0, False),
            "intra_alpha": (0.0, None, False),
            "inter_weight": (0.0, 1.0, False),
            "inter_beta": (0.0, None, False),
        }
        for name, (lower, upper, exclusive) in ranges.items():
            value = _real(name, getattr(self, name))
            if (
                value < lower
                or (exclusive and value == lower)
                or (upper is not None and value > upper)
            ):
                relation = ">" if exclusive else ">="
                suffix = "" if upper is None else f" and <= {upper}"
                raise ValueError(f"chunktrust.{name} must be {relation} {lower}{suffix}")
            object.__setattr__(self, name, value)
        if type(self.inter_use_speed_uniformity) is not bool:
            raise ValueError("chunktrust.inter_use_speed_uniformity must be a boolean")
        for name, choices in (
            ("exec_mode", {"expected_round", "thompson", "qmix_expected_round"}),
            ("update_mode", {"kernel_forget", "independent"}),
            ("intra_z_mode", {"tau0_rms", "window_rms", "trend"}),
            ("mix_mode", {"intra", "inter", "both"}),
            ("inter_fallback", {"ehh_only", "neutral"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"chunktrust.{name} must be one of {sorted(choices)}")
        defaults = {field.name: field.default for field in fields(self)}
        for name, reason in self.inactive_parameters().items():
            if getattr(self, name) != defaults[name]:
                raise ValueError(f"chunktrust.{name} is inactive: {reason}; keep its default")

    @classmethod
    def from_mapping(cls, options: Mapping[str, object]) -> ChunkTrustOptions:
        if not isinstance(options, Mapping):
            raise ValueError("inference.chunktrust must be a mapping")
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown chunktrust parameter(s): {sorted(unknown, key=str)}")
        return cls(**dict(options))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = list(self.candidates)
        return result

    def inactive_parameters(self) -> dict[str, str]:
        """Only certainly inactive controls, never conditional intra fallback paths."""
        inactive: dict[str, str] = {}
        if self.exec_mode == "qmix_expected_round":
            inactive.update(
                dict.fromkeys(
                    (
                        "epsilon",
                        "seed",
                        "update_mode",
                        "kernel_bandwidth",
                        "forget_rho",
                        "update_eta",
                    ),
                    "qmix_expected_round uses neither selector randomness nor posterior updates",
                )
            )
        if self.update_mode == "independent":
            inactive.update(
                dict.fromkeys(
                    ("kernel_bandwidth", "forget_rho", "update_eta"),
                    "independent updates do not use a forgetting kernel",
                )
            )
        if self.exec_mode == "thompson":
            inactive["temperature"] = "thompson selects an argmax without temperature"
        if self.intra_z_mode != "window_rms":
            inactive["intra_tau_window"] = "only window_rms uses the baseline window"
        if self.mix_mode == "intra":
            inactive.update(
                dict.fromkeys(
                    (
                        "inter_use_speed_uniformity",
                        "history_window",
                        "inter_weight",
                        "inter_beta",
                        "inter_fallback",
                    ),
                    "intra mode does not use inter evidence",
                )
            )
        elif self.mix_mode == "inter":
            inactive["inter_weight"] = "only both mode mixes the two evidence terms"
        if not self.inter_use_speed_uniformity:
            inactive.update(
                dict.fromkeys(
                    ("history_window", "inter_beta"),
                    "inter evidence is disabled",
                )
            )
        return inactive

    def parameter_status(self) -> dict[str, dict[str, str]]:
        result = {
            field.name: {"status": "active", "reason": "selected algorithm branch"}
            for field in fields(self)
        }
        for name, reason in self.inactive_parameters().items():
            result[name] = {"status": "inactive", "reason": reason}
        if self.mix_mode != "intra":
            for name in ("history_window", "inter_beta", "inter_fallback"):
                if result[name]["status"] != "inactive":
                    result[name] = {
                        "status": "conditional",
                        "reason": "candidate history feasibility",
                    }
        if self.mix_mode == "inter":
            status = "conditional" if self.inter_fallback == "ehh_only" else "diagnostic_only"
            for name in ("intra_cut_t", "intra_z_mode", "intra_tau_window", "intra_alpha"):
                if result[name]["status"] != "inactive":
                    result[name] = {"status": status, "reason": "intra fallback or diagnostics"}
        return result


def chunktrust_parameters(**options: object) -> dict[str, Any]:
    """Normalize known parameters without changing resolved-config reload semantics."""
    return ChunkTrustOptions.from_mapping(options).to_dict()
