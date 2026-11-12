#!/usr/bin/env python3
"""Recompute the frozen study in a new step-only output directory.

Uses archived runtime sources and assets, immutable historical recordings,
and the frozen development/holdout protocol. No robot or policy startup.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "data/analysis/step_diffik_study/20261019"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    if not dest.is_relative_to(ROOT / "data/analysis/step_diffik_study"):
        parser.error("Output must be inside data/analysis/step_diffik_study")
    dest.mkdir(parents=True, exist_ok=False)
    for name in [
        "joint_factor_space.json",
        "tcp_joint_factor_space.json",
        "frozen_selection.json",
        "frozen_selection_roles.json",
        "rate_contract_audit.json",
    ]:
        shutil.copy2(EVIDENCE / name, dest / name)
    env = dict(
        os.environ,
        STEP_DIFFIK_OUTPUT=str(dest),
        STEP_DIFFIK_SOURCE_ROOT=str(EVIDENCE / "source_snapshot"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    script = Path(__file__).with_name("run.py")
    # Each phase waits for its predecessor; holdout only sees the frozen choices.
    for phase in [
        "baseline",
        "single",
        "rank-single",
        "joint",
        "tcp",
        "tcp-joint",
        "holdout",
        "refresh-metrics",
    ]:
        print("Reproducing", phase, flush=True)
        with (dest / f"{phase}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(script), phase],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    subprocess.run(
        [sys.executable, str(script.with_name("verify.py"))], cwd=ROOT, env=env, check=True
    )
    expected = json.loads((EVIDENCE / "cohort.json").read_text())
    actual = json.loads((dest / "cohort.json").read_text())
    assert expected == actual, "Historical cohort changed; do not compare silently."
    # Plot dependencies were installed only inside the original step evidence directory.
    (dest / "plot-deps").symlink_to(EVIDENCE / "plot-deps", target_is_directory=True)
    subprocess.run(
        [sys.executable, str(script.with_name("report.py"))], cwd=ROOT, env=env, check=True
    )
    print(dest)


if __name__ == "__main__":
    main()
