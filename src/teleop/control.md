# control.md — control commands reference

## bin/

Thin bash wrappers around `scripts/*.py` so the underlying tools can be run by name from any cwd (put `bin/` on `PATH`).  No need to activate the venv first.

### brake-release

```
brake-release A    # 臂 A 强制松闸，按 Enter 抱闸
brake-release B    # 臂 B
```

强制松闸（控制器参数 `BRAK0`/`BRAK1`：2 松闸、1 抱闸，同厂家
`showcase_apply-brake_release-brake.py`）。用于撞机/急停后臂扭成一团上不了使能时，
手动把臂掰回来。**松闸后不上伺服、没有重力补偿，臂会立刻下坠** —— 先托住臂、另一人守急停。
按 Enter 或 Ctrl+C 都会重新抱闸。它不清 err=6：之后进拖动仍要加 `--tool`。

Wraps `scripts/brake_release.py`.

### check-error

```
check-error        # 两臂都清（config.ARMS 默认）
check-error A      # 只清臂 A
check-error AB     # 显式两臂
check-error --retries 10 --wait 0.5
```

只查状态 + 清错：不上伺服、不切模式、不下发任何关节指令，臂不会动。急停拍下后
`err_code=13`(Emcy) 会一直锁着、伺服上不去，松开按钮也不会自己消失 —— 先跑这个
确认两臂都 ✅，再去跑真正要动的脚本。全部清干净才返回 0，可以用 `check-error && go-home`
串起来。

Wraps `scripts/check_errors.py` (see `scripts.md#check_errorspy`).

### get-current-pos

```
get-current-pos
```

Wraps `scripts/get_current_pos.py` (see `scripts.md#get_current_pospy`). No argument parsing of its.

### go-home

```
go-home        # both arms (config.ARMS default)
go-home A      # just arm A
go-home AB     # both arms, explicit
```

Wraps `scripts/home_now.py` (see `scripts.md#home_nowpy`). 

### set-state

```
set-state <A|B> <position|impedance|drag|release|disabled> [extra flags...]

set-state A position
set-state A impedance --impedance-type cartesian
set-state B drag --drag-space X
set-state A drag --tool omnigripper       # + gravity comp for the registered tool
set-state A release --tool omnigripper    # SDK-native zero-force free-drive, also --tool-aware
set-state A disabled
```

`--tool <name>` is already supported -- loads `configs/tool/<name>.yaml` (`drivers.tool_config`) and feeds its mass/COM/inertia into `Marvin_Robot.set_tool()` before entering torque/CR mode.

Wraps `scripts/set_state.py` (see `scripts.md#set_statepy`). 

### tip-calib

```
set-state A release --tool umi     # 终端1：零力拖动
tip-calib --arm A                  # 终端2：只读采点，解算后打印可粘贴的 yaml
```

标定 `configs/tool/umi.yaml` 的 `kine_offset`（法兰→两指中点）；没标定时 UMI 回放
对齐的是法兰而不是指尖（见 `docs/algos.md#tool_framepy`）。纯只读，可与终端 1 并行。

Wraps `scripts/tip_calib.py` (see `scripts.md#tip_calibpy`).

## scripts/

No `bin/` wrapper.

### goto_joints.py

```
python3 scripts/goto_joints.py --arm A --to 86.43,-76.25,-90.94,-84.22,-14.53,1.65,-10.12      # or you can use -to=...
python3 scripts/goto_joints.py --arm A --to ... --speed 5      # slower
python3 scripts/goto_joints.py --arm A --to ... --release      # disable on arrival (locks the pose)
python3 scripts/goto_joints.py --arm A --to ... --control-mode impedance  # compliant tracking, docs/algos.md#impedanceconfig
python3 scripts/goto_joints.py --arm A --to ... --control-mode impedance --impedance-type cartesian
python3 scripts/goto_joints.py --arm A --to ... --viewer        # mirror to universal_viewer, see "Viewer" below
```

### umi_replay.py / umi_filter.py / umi_gripper.py

UMI 数据回放自成一条管线，完整流程（起始构型、预演、夹爪、故障速查）见
[`umi与天机replay.md`](./umi与天机replay.md)。

```
# 离线：回放可视化 / 本体可行性过滤（不连机器人）
python3 replay.py --umi ~/Downloads/replay_test --hand left
python3 scripts/umi_filter.py ~/Downloads/replay_test --arms A --speed 0.5

# 实机：--goto-start 先走到 config.UMI_START_JOINTS
python3 scripts/umi_replay.py ~/Downloads/replay_test --dry-run
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --arms A --speed 0.3
python3 scripts/umi_replay.py ~/Downloads/replay_test --goto-start --gripper

# 夹爪单独验证（不碰机械臂）
python3 drivers/umi_gripper.py --scan
python3 drivers/umi_gripper.py --arm A --num 3
```

`--speed` 是纯时间缩放（0.3 = 慢 3 倍），`--scale` 缩放运动本身，`--frames a:b` 只跑一段。
真实下发前脚本会强制预演整条轨迹并要求确认，别跳过 —— 安全闸没有自碰撞模型。

设计见 `docs/scripts.md#umi_replaypy`、`docs/algos.md#embodimentpy`、`docs/drivers.md#umi_gripperpy`。

### tool_calib_collect.py / tool_calib_identify.py

```
python3 scripts/tool_calib_collect.py --arm B --tool omnigripper
# -> prompts to remove the tool, then moves to all-zero joints and runs the no-load trajectory;
#    prompts to remount the tool, then moves to all-zero joints again and runs the load trajectory
# -> data/tool_calib/omnigripper/B/{NoLoadData.csv,LoadData.csv,CfgFile/...}

python3 scripts/tool_calib_identify.py --dir data/tool_calib/omnigripper/B --arm B --tool omnigripper
# -> prints the identified params + a ready-to-paste block for configs/tool/omnigripper.yaml's
#    arms.B (fill in `source` by hand)
```

Every time a different gripper/tool gets mounted, it needs its own entry in `configs/tool/` (see `scripts.md#tool_calib_collectpy--tool_calib_identifypy`, `config.md#tool`) -- there's no default/fallback tool, and the controller's gravity feedforward silently ignores an unregistered tool's weight rather than erroring, so a swapped gripper without a matching yaml entry won't be caught for you.

## Viewer (`--viewer`)

`goto_joints.py` and `run_teleop.py` both take a `--viewer` flag that mirrors live joint state into `universal_viewer`'s browser dashboard. 
```
# terminal 1, in universal_viewer/
uv run universal-policy-viewer --robot tianji --port 8086   # open http://localhost:8086

# terminal 2, in teleop/
python3 scripts/goto_joints.py --arm A --to ... --viewer
python3 run_teleop.py --arms A --viewer
```

Both arms are always published, not just whichever one is being driven. `goto_joints.py` only updates the dashboard before the move, after it lands, and during the post-arrival hold loop. Full design rationale: `docs/viewer.md`.
