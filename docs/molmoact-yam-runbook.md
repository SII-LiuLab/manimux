# MolmoAct + YAM 运行手册

运行前清空机械臂工作区并准备好急停。以下四个服务分别在四个终端中按顺序启动。

## 配置位置

```text
ManiMux: manimux/configs/experiments/pick_red_object/yam_molmoact2_manimux.yaml
RTC:     manimux/configs/experiments/pick_red_object/yam_molmoact2_rtc.yaml
```

以后增加其他本体时放在 `configs/molmoact2/<embodiment>/`，不要再创建顶层扁平 YAML。

## 1. MolmoAct 模型服务

```bash
cd /home/ubuntu/manimux
source envs/yam/.venv/bin/activate
manimux-molmoact-server \
  --host 127.0.0.1 \
  --port 8202 \
  --repo-id allenai/MolmoAct2-BimanualYAM \
  --device cuda:0 \
  --dtype bfloat16
```

看到 `Warmup OK` 和 `Uvicorn running` 后继续。

## 2. RealSense 相机服务

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config manimux/configs/embodiment/sensor/cameras/yam.yaml
```

确认三台相机均已打开，并看到 `REP bound` 和 `PUB bound`。

## 3. Viewer

```bash
cd /home/ubuntu/manimux
source .venv/bin/activate
manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

浏览器打开 `http://localhost:8086`。

## 4. ManiMux 真机 runtime

先确认 `can_left` 和 `can_right` 都是 `ERROR-ACTIVE`（检查和重开命令见
[CAN 总线](can-bus.md)）：

```bash
for c in can_left can_right; do printf '%s: ' "$c"; ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'; done
```执行后机械臂会按配置的
默认配置使用 `100 Hz` 控制环，并用 `start_duration_s=3.0` 移动到起始姿态，
随后保持 `PAUSED`；在 Viewer 确认轨迹后再点
`Start rollout`；暂停后使用 `Resume rollout`。

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config manimux/configs/experiments/pick_red_object/yam_molmoact2_manimux.yaml
```

真机速度和时间直接修改 `manimux/configs/experiments/pick_red_object/yam_molmoact2_manimux.yaml`：

- `policy.action_dt_s`：相邻 Policy 轨迹点的时间间隔（秒）；
- `executor.smooth.max_velocity`：关节最大速度（rad/s）；
- `executor.smooth.max_acceleration`：关节最大加速度（rad/s²）；
- `robot.options.start_duration_s` / `home_duration_s`：起始姿态和回零秒数。

当前 ManiMux infra 配置使用老师原 ManiMux 的 `action_dt_s: 0.05`；30 个轨迹点约覆盖
1.45 秒。关节限制为 `0.25 rad/s`、`0.5 rad/s²`。首次恢复真机时仍应只做
一次短距离、工作区清空且急停就绪的低速验证。CAN 处于 `ERROR-PASSIVE` 或
`BUS-OFF` 时禁止启动。

## 停止

启动 runtime 后会自动推理和执行，不需要在 Viewer 中点击 Start。达到
`run.max_control_steps` 后，机械臂自动回 Home，然后 runtime 退出。

正常运行时在 runtime 终端按一次 `Ctrl-C`，等待机械臂回零并退出；随后再停止
相机、模型服务和 Viewer。

新 rollout 保存在 config 指定的 `data/experiments/.../session-*/rollout-*`；未完整结束的记录保留
`.partial` 后缀。2026-08-24 之前散落在 `data/run-*` 的探索记录已归档到
`data/archive/pre-campaign-20260824/root-runs/`。

## Chunk 边界诊断

普通 runtime 会在每个新 plan 接受时写一条 `plan_boundary` 到 episode 的
`events.jsonl`。它同时保存模型原始 chunk 首点、ManiMux 拼接后的首点、上一条命令和实测
关节；只增加诊断记录，不改变动作、速度或拼接算法。即使按 `Ctrl-C`，记录也会保留在
`.partial` episode 中。

分别完成 MolmoAct2 和 Pi05 的短 rollout 后，只需告知“跑完了”；分析端会自动选取两种
policy 最新的有效 episode。手动离线分析命令为：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/analyze_chunk_boundaries.py
```
