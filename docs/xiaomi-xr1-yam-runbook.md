# Xiaomi XR-1 + YAM 运行手册

本文只覆盖 XPolicyLab 的 `Xiaomi_Robotics_1` 路线。XPolicyLab 已集成
[小米官方 XR-1 源码](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)，用户不需要
再 clone 一份官方仓库。这个 adapter 由我们的 XPolicyLab fork 维护，不是上游
XPolicyLab 原本自带；模型加载、预处理和 denoise 仍完整运行在 XPolicy 标准 server
内。ManiMux 只负责 wire codec、YAM FK/IK 和执行，不提供平行的 native model server。

该链路运行时，机械臂收到的是官方 `Xiaomi-Robotics-1-5B` 经过完整 forward 和 denoise
产生的动作，不是启动姿态、预录轨迹或 mock。XPolicy 负责真实模型推理；ManiMux 负责将
原生 EE delta 转为 YAM joint position 并调度执行。

## 当前权重边界

本地 `model_states.pt` 来自官方
[`Xiaomi-Robotics-1-5B`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B)。
官方 model card 将它定位为继续 post-training 的起点，不是 YAM 策略。
官方另外发布的 RoboCasa / RoboCasa365 / VLABench 权重也都是特定仿真本体，
不能直接驱动双臂 YAM。

## 动作链路

```text
三路 RGB + 14-D YAM state
  -> XPolicy Xiaomi_Robotics_1
  -> 30 x 60 anchor-relative EE deltas
  -> ManiMux xr1_yam adapter FK/IK
  -> 30 x 14 absolute YAM joints
```

模型原生输出不是 joint position。每一步是相对请求时末端锚点的位置、轴角和夹爪增量；
YAM FK/IK 只存在于 ManiMux adapter 中。

官方公开的 post-training 数据格式没有声明控制 Hz。当前 `action_dt_s=0.033333`
（30 Hz）是 YAM 部署假设，不是已从 XR-1 checkpoint 证明的训练频率。未来 YAM
fine-tune bundle 必须记录原生采样 Hz，并用同一值替换该配置。

## 模型环境

XR-1 模型服务使用 `envs/xr1/.venv`。切换到 XPolicy server 后，该环境不仅需要模型
依赖，还必须安装 XPolicyLab 的 HDF5、WebSocket 和 msgpack 依赖：

```bash
cd /home/ubuntu/manimux
uv pip install --python envs/xr1/.venv/bin/python -e XPolicyLab
```

缺少这一步时，服务会在导入 `XPolicyLab.utils.process_data` 时首先报
`ModuleNotFoundError: h5py`，继续手工补单个包还会遗漏后续 wire 依赖。

## 配置

```text
base server:    configs/xiaomi-xr1/yam/server/base.yaml
ManiMux:        configs/xiaomi-xr1/yam/infra/manimux.yaml
RTC:            configs/xiaomi-xr1/yam/infra/rtc.yaml
step-15000 RTC: configs/xiaomi-xr1/yam/infra/rtc-assemble-screwdriver-step15000.yaml
```

RTC 将 ManiMux `30 x 14` overlap condition 通过 FK 反编码到模型原生 `30 x 60` 空间，
再进入五步 flow denoise 的 PiGDM conditioning。这是 ManiMux/XPolicy 扩展，
不是小米官方 XR-1 部署功能；它通过明确 sampler hook 实现，不是运行时
monkeypatch，也没有额外动作幅度阈值。

完整 RTC 链路是：

```text
旧的 30 x 14 absolute joint tail + soft mask
  -> 按新 observation 的 FK 锚点重新编码为 30 x 60 EE delta
  -> 用 action mean/std 进入模型归一化空间
  -> XR-1 每一步 Euler denoise 计算 clean estimate 与 VJP guidance
  -> 反归一化 -> FK/IK -> 新的 30 x 14 joint chunk
```

因此它是 sampler-level RTC，不是 chunk splice。但目前只有 CPU 虚拟 flow 证明
guidance 确实在 `_generate` 内生效；还没有 5B 模型的真实 GPU conditioned
forward，所以 I7 仍然只能标记为离线完成。

`yam.json` 已经按官方格式由 `60` 个完整 YAM episode、`23,994` 个 30-step window
计算：action 是 `30 x 60` anchor-relative EE delta 的 mean/std，state 是 `1 x 60`
YAM joint/FK state 的 q01/q99。因此 **base 测试不需要再改数值**；改用官方 washer
demo stats 反而会把另一台机器的单位送给 YAM。

但这份 stats 只让 YAM state/action 映射在数值上有定义，不是官方 5B checkpoint 的
配对 post-training statistics。只有用同一份 YAM 数据 fine-tune 并导出权重后，二者才
真正匹配。重新采集数据或改变 action codec 时再运行：

```bash
cd /home/ubuntu/manimux
PYTHONPATH=src envs/yam/.venv/bin/python -m \
  manimux.integrations.xr1_yam.compute_norm_stats \
  --episodes /path/to/yam/episodes \
  --out src/manimux/integrations/xr1_yam/norm_stats/yam.json
```

## Base 权重能力测试

base 权重使用独立 server config，不覆盖未来的 YAM finetune；执行仍复用标准 ManiMux：

```bash
# offline contract check
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/servers/xiaomi_xr1_yam_server.py \
  --config configs/xiaomi-xr1/yam/server/base.yaml \
  --check

# terminal 1: model server
envs/xr1/.venv/bin/python scripts/servers/xiaomi_xr1_yam_server.py \
  --config configs/xiaomi-xr1/yam/server/base.yaml

# terminal 2: no-CAN GPU/WS/FK/IK probe
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/xiaomi-xr1/yam/infra/manimux.yaml
```

probe 必须返回有限的 `native_shape: [30, 60]` 与 `canonical_shape: [30, 14]`。
通过后再启动相机，并由操作者运行：

```bash
envs/yam/.venv/bin/manimux run \
  --config configs/xiaomi-xr1/yam/infra/manimux.yaml
```

30 Hz 仍是 YAM 对照实验假设，不是官方 checkpoint 元数据。base 能否做任务是
policy 实验结果；shape、finite、WS 和 FK/IK 才是本轮 infra 验收项。

2026-08-20 已在 RTX 4090 完成 base GPU/WS/FK/IK probe。模型加载 `1135` 个 tensor，
原生输出为有限的 `30 x 60`，转换后为有限的 `30 x 14` absolute joint chunk。第一次
冷请求为 `953.0 ms`，随后三次热态为 `164.5 / 151.7 / 152.5 ms`。30 步在 30 Hz 下
覆盖约 `1.0 s`，热态推理约占 `4.6-5.0` 步，默认 ManiMux 具备供给余量。该证据不包含
相机、CAN 或机械臂，也不证明 base checkpoint 能完成 YAM 任务。

## 真机运行：Base + ManiMux

首次只运行默认 ManiMux，不使用 RTC。清空双臂工作区并准备急停；不要同时启动第二个
XR-1 server 或 RTC runtime。

### Terminal 1：XR-1 XPolicy 模型服务

```bash
cd /home/ubuntu/manimux
envs/xr1/.venv/bin/python scripts/servers/xiaomi_xr1_yam_server.py \
  --config configs/xiaomi-xr1/yam/server/base.yaml
```

看到模型加载完成并监听 `127.0.0.1:8500` 后，先在另一个终端完成无 CAN probe：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/xiaomi-xr1/yam/infra/manimux.yaml
```

只有 probe 返回有限的 `native_shape: [30, 60]` 和
`canonical_shape: [30, 14]` 才继续。

### Terminal 2：三相机服务

已有 `5555` 相机服务时不要重复启动：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml
```

### Terminal 3：Viewer

```bash
cd /home/ubuntu/manimux
.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
```

### Terminal 4：CAN 检查与 ManiMux

```bash
for c in can_left can_right; do
  printf '%s: ' "$c"
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

只有两路均为 `ERROR-ACTIVE` 才运行：

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/xiaomi-xr1/yam/infra/manimux.yaml
```

连接后的前 `5.0 s` 是配置规定的起始姿态移动，不是模型动作；之后才执行 XR-1 经
XPolicy 输出、再由 YAM FK/IK 转换得到的关节命令。正常停止时只在 runtime 终端按一次
`Ctrl-C`，等待 `5.0 s` 回 Home 和 Recorder 收尾，再停止相机、模型服务和 Viewer。

## RTC 对照

先用离线脚本验证 XPolicy 使用的 sampler guidance hook：

```bash
envs/xr1/.venv/bin/python scripts/validation/check_xr1_rtc_sampler.py
```

它不构造 5B 模型、不访问 GPU。只有 base server 的 ManiMux forward 和默认 runtime
通过后，才使用同一个 server 做 RTC 对照：

```bash
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/manimux-assemble-screwdriver-step15000.yaml
envs/yam/.venv/bin/manimux run --config configs/xiaomi-xr1/yam/infra/rtc-assemble-screwdriver-step15000.yaml
```

不要同时运行 ManiMux 与 RTC。相机、Viewer、CAN 检查和停止顺序参考
[MolmoAct + YAM](molmoact-yam-runbook.md)。

## 当前边界

已验证配置、权重/processor/stats 路径、GPU normal forward、XPolicy WS、动作 codec、
FK/IK normal action conversion、RTC condition round-trip、payload 和两条 sampler
guidance 分支。尚未验证真实 GPU conditioned RTC forward；真实相机/CAN/真机链路已经
启动过，但动作语义验收失败。`model_states.pt` 是 post-training 起点，不是已经证明能在
YAM 上完成任务的策略。

2026-08-20 的首次 YAM 闭环中，左臂出现绕向本体背面的动作。该结果将真机 Gate 判定为
**未通过**：当前证据只证明 60D 输出能被转换成有限的 14D joint chunk，不证明模型 EE
坐标、左右臂映射、IK 分支和 YAM 关节语义正确。在完成 Recorder 离线回放、逐步 EE 目标
重建、左右臂坐标系对照和 joint-limit 审计前，不再运行该配置的真机。
