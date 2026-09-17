# tianji-control

## ManiMux source snapshot

This standalone project is stored in `src/teleop/`. It is not wired into
ManiMux's runtime, collection, entry points or package dependencies. Run its
commands from this directory and manage its environment with its own
`pyproject.toml` and `uv.lock`.

The source snapshot includes the local `sine_probe` script, offline tests and
documentation. Git history, virtual environments, caches, generated URDF assets,
logs, benchmark results, recorded trajectories (`data/traj_*.jsonl`) and local
start poses (`data/dp_start_*.yaml`) are not included. Existing examples referring
to those data files require local copies; `scripts/goto_dp_start.py` needs a local
pose file, selectable with `--pose`. External SDK and viewer paths in the original
configuration/documentation remain installation-specific.

Dual-arm teleoperation stack for the Tianji robot, driven from a PICO XR
headset/controllers.

## Layout

- `run_teleop.py` — main entry point, wires XR input → IK solver → arm driver
- `replay.py` — replay recorded trajectories offline (XR recordings, and UMI episodes via `--umi`, whole or `--frames A:B`)
- `config.py` — central runtime configuration
- `algos/` — IK solvers, null-space control, retargeting, safety gate,
  tool/fingertip frame, embodiment feasibility filter
- `core/` — session orchestration (`ArmChannel`, homing)
- `drivers/` — arm SDK wrapper, gripper drivers (OmniGripper + UMI), XR and UMI data sources
- `scripts/` — standalone calibration/debug/bring-up CLI tools
- `bin/` — thin bash wrappers around `scripts/*.py` for running by name from any cwd
- `configs/` — YAML tuning files (solver, tool)
- `docs/` — design notes and rationale behind `config.py` and each module (see `docs/README.md`)
- `bench/` — IK solver benchmarking

See the `run_teleop.py` docstring and `docs/README.md` for more detail.

## Getting started

- New to this repo and about to run teleop on real hardware? Start with
  [`快速上手-PICO与天机Teleop.md`](./快速上手-PICO与天机Teleop.md) — step-by-step
  PICO + Tianji bring-up checklist.
- Need to drive the arms directly (homing, state switching, joint moves)?
  See [`control.md`](./control.md) for the `bin/`/`scripts/` command reference.
- Replaying a UMI demo (handheld TacCap rig) on the arms? See
  [`umi与天机replay.md`](./umi与天机replay.md) — data format, offline
  verification, start-pose selection, and the on-robot ladder.
