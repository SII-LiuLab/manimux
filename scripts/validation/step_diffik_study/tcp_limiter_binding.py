"""Attribute the active pose-limit fraction on the fixed pi/80 baseline."""

import json

import numpy as np
import run as study
from check_tcp_speed import BASE, OUT
from reference_layer_review import fk_batch
from scipy.spatial.transform import Rotation


def main():
    rows = [r for r in json.loads((study.OUT / "cohort.json").read_text()) if r["included"]]
    results = []
    for row in rows:
        d = study.load(row)
        model = study.make_model(d["c"])
        trace = dict(np.load(BASE / row["id"] / "legacy_tcp_pi80/trace.npz"))
        active = trace["reason"] != "gap"
        dt = 1 / d["c"]["robot"]["control_hz"]
        previous = np.concatenate([trace["command"][:1], trace["command"][:-1]])
        for j, g in enumerate(study.GROUPS):
            matrices = fk_batch(model.kinematics.models[g], previous[active, j])
            target = d["target"][active, j]
            distance = np.linalg.norm(target[:, :3] - matrices[:, :3, 3], axis=1)
            rotations = Rotation.from_quat(target[:, [4, 5, 6, 3]]).as_matrix()
            angle = Rotation.from_matrix(
                rotations @ np.swapaxes(matrices[:, :3, :3], -1, -2)
            ).magnitude()
            linear = 0.4 * dt / np.maximum(distance, 1e-15)
            angular = 1.8 * dt / np.maximum(angle, 1e-15)
            fraction = np.minimum(1, np.minimum(linear, angular))
            error = float(np.max(abs(fraction - trace["fraction"][active, j])))
            assert error < 1e-6, error
            results.append(
                {
                    "episode": row["id"],
                    "task": row["task"],
                    "split": row["split"],
                    "status": row["status"],
                    "stratum": d["info"]["stratum"],
                    "arm": g,
                    "active_ticks": int(active.sum()),
                    "linear_binding": int(((linear < 1) & (linear < angular)).sum()),
                    "angular_binding": int(((angular < 1) & (angular <= linear)).sum()),
                    "unlimited": int((fraction >= 1).sum()),
                    "fraction_reconstruction_max_error": error,
                }
            )
    study.save(OUT / "limiter_binding.json", results)
    print("pose fractions verified on", len(results), "episode-arms")


if __name__ == "__main__":
    main()
