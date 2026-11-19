"""Deadline-bound paired CUDA refinement for the frozen Tianji step policies."""

from collections import deque
from time import monotonic_ns

import numpy as np

from manimux.kinematics.poses import pose_matrix
from manimux.runtime.rl_step import ROOT

ARMS = ("left_arm", "right_arm")


class GPURefiner:
    def __init__(self, options, *, fixed=0):
        self.fixed = fixed
        if set(options) != {"python", "merge_reserve_us", *ARMS}:
            raise ValueError("CUDA RL IK requires python, merge_reserve_us and both exports")
        self.reserve_ns = int(options["merge_reserve_us"] * 1000)
        if not 0 < self.reserve_ns < 2_000_000:
            raise ValueError("invalid paired CUDA merge reserve")
        self.options = dict(options, python=str(ROOT / options["python"]))
        for arm in ARMS:
            spec = options[arm]
            self.options[arm] = dict(spec, actor=str(ROOT / spec["actor"]))
        self.pair = None
        self.history = deque(maxlen=8)
        self.sequence = 0
        self.driver_error = None

    def warmup(self, models, seed):
        if set(models) != set(ARMS) or set(seed) != set(ARMS):
            raise ValueError("paired CUDA IK requires left_arm and right_arm")
        if self.pair is None:
            from rl_ik.step_cuda import CudaPair

            self.pair = CudaPair(self.options)
            try:
                for i, arm in enumerate(ARMS):
                    q = np.r_[self.pair.verify_q[i], seed[arm][7]]
                    if not np.allclose(
                        models[arm].fk(q), self.pair.verify_fk[i], atol=1e-7, rtol=0
                    ):
                        raise ValueError(f"{arm}: CUDA and station TCP disagree")
            except BaseException:
                self.close()
                raise
            self.meta = self.pair.meta
            self.driver_error = None
        self.history.clear()

    def propose(self, ref, seed, measured, deadline):
        self.sequence += 1
        start = monotonic_ns()
        past = list(self.history)[-2::-2][::-1]
        self.history.append({arm: seed[arm][:7].copy() for arm in ARMS})
        candidate, count, reason = None, 0, "cuda_error"
        if self.driver_error is None:
            try:
                candidate, count, reason = self._refine(ref, seed, measured, past, deadline)
            except RuntimeError as exc:
                # A driver failure invalidates the stream. Brake until reinitialized.
                self.driver_error = str(exc)
        end = monotonic_ns()
        expired = end >= deadline
        valid = (
            candidate is not None
            and count > 0
            and not expired
            and candidate.shape == (2, 7)
            and np.isfinite(candidate).all()
        )
        stats = {
            arm: dict(
                iterations=count,
                timeout=expired
                or (candidate is None and reason not in {"invalid_target", "cuda_error"}),
                elapsed_ms=(end - start) / 1e6,
                state=reason,
                sequence=self.sequence,
                **({"error": self.driver_error} if self.driver_error else {}),
            )
            for arm in ARMS
        }
        return (dict(zip(ARMS, candidate, strict=True)) if valid else {}), stats, {}

    def _refine(self, ref, seed, measured, past, deadline):
        # A rejected graph may still read pinned input or write its staging output.
        with self.pair.active():
            if not self.pair.complete():
                return None, 0, "previous_graph_running"
            packet = self.pair.input
            packet.fill(0)
            for index, arm in enumerate(ARMS):
                row = ref.groups[arm][0]
                if not np.isfinite(row).all() or not 0 <= row[7] <= 1:
                    return None, 0, "invalid_target"
                x = packet[index]
                x[28:35] = seed[arm][:7]
                x[46] = measured[arm][7]
                x[47:63] = pose_matrix(row[:7]).ravel()
                if past:
                    x[35:42] = x[28:35] - past[-1][arm]
                    x[46 - len(past) : 46] = 1
                    for j, item in enumerate(past, 4 - len(past)):
                        x[7 * j : 7 * j + 7] = item[arm]
            return self.pair.infer(deadline, self.reserve_ns, self.fixed)

    def finish_telemetry(self, wait_s):
        # Graphs discarded after the cut-off never generate a command or an IPC reply.
        return ()

    def close(self):
        if self.pair is not None:
            self.pair.close()
            self.pair = None
