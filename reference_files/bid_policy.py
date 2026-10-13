"""B5: BID backward coherence baseline.

Samples M candidates with shared KV cache, selects the one most temporally
coherent with the previously executed chunk (lowest weighted L2 over overlap).
"""

import numpy as np
from openpi_client import base_policy as _base_policy


class BIDBackwardPolicy(_base_policy.BasePolicy):
    def __init__(self, base_policy, sampler, M: int = 16, rho: float = 0.9,
                 action_horizon: int = 50, replan_steps: int = 5):
        self.base_policy = base_policy
        self.sampler = sampler
        self.M = M
        self.rho = rho
        self.H = action_horizon
        self.replan_steps = replan_steps
        self.prev_chunk = None

    def infer(self, obs: dict) -> dict:
        if obs.get("episode_start", False):
            self.prev_chunk = None

        result = self.sampler.sample(self.base_policy, obs, M=self.M)
        candidates = result["actions_full"]  # (M, H, 7)

        if self.prev_chunk is None:
            best_idx = 0
        else:
            overlap = self.H - self.replan_steps
            scores = np.zeros(self.M)
            for m in range(self.M):
                for tau in range(overlap):
                    t_prev = tau + self.replan_steps
                    if t_prev < self.H:
                        diff = np.linalg.norm(
                            candidates[m, tau] - self.prev_chunk[t_prev]
                        )
                        scores[m] += (self.rho ** tau) * diff
            best_idx = int(np.argmin(scores))

        chosen = candidates[best_idx]
        self.prev_chunk = chosen.copy()

        out = dict(result["base_result"])
        out["actions"] = chosen
        return out

    def reset(self):
        self.prev_chunk = None
