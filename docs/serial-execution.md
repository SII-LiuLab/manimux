# 固定前缀 chunk 的纯串行执行

放瓶子 Pi05 joint-step30000 使用：

```bash
envs/yam/.venv/bin/manimux serve \
  --config configs/pi05/yam/infra/put-bottles/serial-joint-step30000.yaml
```

```yaml
execution:
  runtime: manimux
  inference_schedule: serial
  chunk_steps: 12
  commit_lead_s: 0.0
  blend_steps: 0
```

流程为 **新观测 → 推理完整 chunk → 执行前 12 步 → 新观测 → 推理**。不预取，不在
执行过程中启动下一次推理。推理仍在 worker 中完成，控制线程继续发送保持
指令并处理 GUI Pause/Finish；这里的串行是任务时序，不是阻塞控制线程。

轨迹从结果提交时开始计时，不按推理延迟跳过前几行。Pi05 模型输出 contract
仍是 50 步，OpenWAM 仍是 32 步；`execution.chunk_steps: 12` 只在 timeline
commit 边界保留原始前 12 行。
记录保留真实观测时间，
`max_plan_age_s` 仍限制过期结果，不能用重设观测时间掩盖延迟。每行按模型
30 Hz 时钟计时，最后一行保留一个完整动作周期，整段约 12/30 = 0.40 秒。
整段结束后才开始下一次推理，等待期间保持最后一次下发的关节/夹爪指令。
按 Pause 会清掉当前轨迹并丢弃之前的在途结果；再次 Start 后采集新观测推理。

保留默认放瓶子配置的 smooth、关节/夹爪限速、相机和模型身份，取消 chunk
边界混合。50 步指轨迹时间窗口，受限速和跟踪误差影响，不能保证机械臂在窗口
结束时恰好到达预测终点；这项实现没有增加“实际到位再重规划”的条件。

`refill_threshold_s` 不适用于串行模式，填写会报错。目前串行配置要求 inline
解码，并拒绝已被 adapter 裁掉开头的 chunk。Pi05 joint 输出无需 IK；OpenWAM
的 EEF 输出仍需在 embodiment adapter 中转换为 joint。

验证覆盖延迟后从提交时执行固定前缀、最后一个动作周期、段间固定保持、
等待期间不重复请求、暂停丢弃旧结果，以及旧异步时间轴的回归；未启动真机。
