"""Execution policy for unsuccessful IK; mathematical solvers remain policy-free."""

import math

import numpy as np

FAILURE_POLICIES = frozenset({"hold_last_valid", "reject"})
DEFAULT_FAILURE_POLICY = "hold_last_valid"


def validate_failure_policy(value):
    if not isinstance(value, str) or value not in FAILURE_POLICIES:
        raise ValueError(f"execution.ik.failure_policy must be one of {sorted(FAILURE_POLICIES)}")
    return value


class IKConvergenceError(ValueError):
    """A valid target could not be solved; malformed inputs remain ordinary errors."""


def failed_ik_command(previous, *, policy, reason):
    """Preserve the complete arm/tool command, never the unsuccessful solver candidate."""
    validate_failure_policy(policy)
    if policy == "reject":
        raise IKConvergenceError(f"IK failed: {reason}")
    return np.array(previous, dtype=float, copy=True)


class IKSequence:
    """One group's decoded trajectory, including its trailing tool coordinate.

    Failure holds exactly one output time step and retains the valid seed. The
    next target is still attempted. Metadata records substitutions for experiments.
    """

    def __init__(self, seed, *, group, policy=DEFAULT_FAILURE_POLICY, approximate=False):
        self._current = np.array(seed, dtype=float, copy=True)
        if (
            self._current.ndim != 1
            or len(self._current) < 2
            or not np.isfinite(self._current).all()
        ):
            raise ValueError("IK seed must be a finite arm/tool vector")
        self.group = group
        self.policy = validate_failure_policy(policy)
        self.steps = 0
        self.failures = []
        # approximate: the solver may accept residuals (DLS IK); record them.
        self.approximate = approximate
        self.approximations = []
        self.last_failed = False

    @property
    def current(self):
        # A solver may mutate its seed; the accepted fallback must remain intact.
        return self._current.copy()

    def resolve(self, converged, candidate, *, reason="no_solution", residual=None):
        """residual marks an accepted approximate pose that missed the strict tolerance."""
        self.last_failed = not converged
        index = self.steps
        self.steps += 1
        if converged and residual is not None:
            self.approximations.append({"step": index, **residual})
        if not converged:
            self._current = failed_ik_command(
                self._current,
                policy=self.policy,
                reason=f"group={self.group} step={index} reason={reason}",
            )
            self.failures.append({"step": index, "reason": reason})
        else:
            values = np.asarray(candidate, dtype=float)
            if values.shape != self._current.shape or not np.isfinite(values).all():
                raise ValueError(f"IK invalid solution: group={self.group} step={index}")
            self._current = values.copy()
        return self._current.copy()

    def solve(self, callback):
        """Use for solvers with internal interpolation; never swallow unrelated errors."""
        try:
            candidate = callback()
        except IKConvergenceError as exc:
            return self.resolve(False, None, reason=str(exc))
        return self.resolve(True, candidate)

    def solve_interpolated(
        self, callback, *, duration_s, substep_dt_s, substep_joints=None, reset=None
    ):
        """Accept a whole knot or discard all partial substeps and solver history."""
        try:
            command = self.solve(callback)
        finally:
            if self.last_failed and reset is not None:
                reset()
        if self.last_failed:
            if substep_joints is not None:
                count = max(1, math.ceil(duration_s / substep_dt_s))
                substep_joints[:] = [command[:-1].copy() for _ in range(count)]
        return command

    def metadata(self):
        return {
            "failure_policy": self.policy,
            "solved_steps": self.steps - len(self.failures),
            "failed_steps": len(self.failures),
            "failed_indices": [item["step"] for item in self.failures],
            "failures": list(self.failures),
            **({"approximate_steps": len(self.approximations),
                "approximations": list(self.approximations)}
               if self.approximate else {}),
        }
