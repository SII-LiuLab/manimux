# docs/ — design notes index

Rationale, empirical measurements, and incident history behind the terse
in-code comments. Code says *what*; these say *why*. One file per layer:

- [`algos.md`](algos.md) — IK, null-space control, retargeting, safety gate,
  tool/fingertip frame, embodiment filter
- [`drivers.md`](drivers.md) — arm SDK wrapper, gripper drivers, XR and UMI data sources
- [`core.md`](core.md) — session orchestration (`ArmChannel`, homing), `run_teleop.py`, `replay.py`
- [`config.md`](config.md) — calibration history and tuning rationale behind every constant in `config.py`
- [`scripts.md`](scripts.md) — standalone calibration/debug tools in `scripts/`
- [`viewer.md`](viewer.md) — optional `--viewer` mirror to universal_viewer (`viewer_bridge.py`)

操作手册（不是设计笔记）在仓库根目录：
[`快速上手-PICO与天机Teleop.md`](../快速上手-PICO与天机Teleop.md)（遥操作）、
[`umi与天机replay.md`](../umi与天机replay.md)（UMI 回放）、
[`control.md`](../control.md)（命令参考）。

See the top-level `run_teleop.py` docstring for the overall code layout.
