# ManiUniCon Meshcat simulation

ManiMux treats ManiUniCon as an optional robot adapter, not a core dependency.
The adapter composes two `MeshcatInterface` instances into one canonical dual-arm
robot with these groups:

```text
left_arm, right_arm, left_gripper, right_gripper
```

## Configuration

Copy the machine-local file and edit the ManiUniCon checkout path:

```bash
cp configs/robots/maniunicon_meshcat_dual.example.yaml \
  configs/robots/maniunicon_meshcat_dual.yaml
```

The destination is gitignored. Then run:

```bash
uv run python scripts/validation/run_headless.py --config configs/maniunicon_meshcat.example.yaml
uv run python scripts/validation/run_headless.py --config configs/maniunicon_meshcat.example.yaml --executor mpc
```

ManiUniCon currently imports Torch eagerly at package import time even though the
Meshcat path does not use it. Keep its simulator dependencies in a separate
environment; do not add them to ManiMux's core `pyproject.toml`.

## `jw` smoke result

Validated on 2026-08-16 with two XArm6 Meshcat interfaces and one fake local
policy worker:

| Executor | Ticks | Accepted | Rejected | Mean period | P95 period |
|---|---:|---:|---:|---:|---:|
| Smooth | 120 | 4 | 0 | 20.000 ms | 20.281 ms |
| MPC | 120 | 5 | 0 | 20.003 ms | 20.285 ms |

Both runs finalized Zarr/JSONL episodes containing raw plans and the
scheduled/optimized/command/measured-state lineage. This validates software
composition and timing only; Meshcat is not a dynamics or hardware-safety test.
