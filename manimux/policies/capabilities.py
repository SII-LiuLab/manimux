from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PolicyCapabilities:
    """Capabilities of the live policy backend, reported after startup."""

    sampling_modes: frozenset[str] = frozenset({"default"})
    backend_metadata: dict[str, object] = field(default_factory=dict)

    def supports(self, sampling_mode: str) -> bool:
        return sampling_mode in self.sampling_modes


def metadata_mismatches(
    expected: dict[str, object],
    actual: dict[str, object],
    *,
    path: str = "backend",
) -> list[str]:
    """Compare configured identity fields; extra backend metadata is allowed."""
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in actual:
            mismatches.append(f"{field_path} is missing (expected {expected_value!r})")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                mismatches.append(
                    f"{field_path} expected a mapping, got {type(actual_value).__name__}"
                )
                continue
            mismatches.extend(metadata_mismatches(expected_value, actual_value, path=field_path))
            continue
        if actual_value != expected_value:
            mismatches.append(f"{field_path} expected {expected_value!r}, got {actual_value!r}")
    return mismatches
