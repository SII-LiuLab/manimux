# data/tool_calib/ — 工具参数标定数据

每个工具一个子目录：`<tool>/<arm>/`，内含
`LoadData.csv` / `NoLoadData.csv` / `CfgFile/LoadIdenCfg_Marvin_CCS.txt`，
由 `scripts/tool_calib_collect.py` 生成，`scripts/tool_calib_identify.py` 消费。

> **CSV 只保存在采集机本地，不进 git**（见 `.gitignore`）。repo 里只跟踪
> `CfgFile/` 和本 README；辨识结果记录在 `configs/tool/*.yaml`。换机器或重新
> clone 后要重算，需要从采集机拷回对应的 CSV。2026-09-15 之前的 commit 里仍保留着
> `umi/`、`umi_new/`、`omnigripper/` 当时的 CSV，可用 `git show <commit>:<path>` 取回。

## 目录说明（务必先读）

- **`omnigripper/` — Tianji 原装 OmniGripper（DM4310）标定数据，勿覆盖。**
  2026-08-23 已整体备份到 `data/tool_calib_backup/omnigripper_20260823_213935/`
  （含 `configs/tool/omnigripper.yaml`）。之后任何新的 Omnigripper 采集都应写入
  新目录或先再备份一次，不要直接覆盖。
- **`umi/` — UMI 夹爪标定数据（2026-08-23 起）。**
  当前左臂（臂 A）安装的是 **UMI 夹爪，不是 Tianji/OmniGripper**，因此本次
  校准用 `--tool umi`，输出到 `umi/A/`，与 `omnigripper/` 完全隔离。

## 当前校准会话（2026-08-23）

- 臂：A（左臂）
- 工具：UMI 夹爪（`--tool umi`）
- 状态：**已完成**。采集于 `umi/A/`，辨识结果已写入
  `configs/tool/umi.yaml` 的 `arms.A`（mass 0.759 kg）。

### 右臂 B（同日追加）

- 状态：**已完成**。采集于 `umi/B/`，辨识结果已写入
  `configs/tool/umi.yaml` 的 `arms.B`（mass 0.725 kg）。

> **2026-08-23 重校**：首次 B 臂辨识（0.725 kg）在 release 模式验证时严重下垂
> （关节 4 漂到 -130°），怀疑带载采集时夹爪未装到位。重新采集后新值为
> **0.746 kg / COM [-26.0, -2.8, 82.1] mm**，与 A 臂（0.759 kg）更接近，
> 已更新 `configs/tool/umi.yaml` 的 `arms.B`。首次数据备份于
> `data/tool_calib_backup/umi_B_first_20260823/`。

> **2026-08-23 第三次采集（当前值）**：E-stop 事件后重新采集，辨识结果为
> **0.739 kg / COM [-31.7, -7.6, 82.7] mm**。质量跨三次稳定（0.725/0.746/0.739），
> 质心随每次安装浮动（x -26..-32，y -3..-8，z 82..88），说明安装姿态对辨识
> 影响显著。第二次数备份于 `data/tool_calib_backup/umi_B_second_20260823/`。
