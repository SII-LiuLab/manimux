# 统一执行步数配置

用 `execution.chunk_steps` 调整各算法的执行窗口/重推理间隔：

```yaml
execution:
  runtime: paint  # manimux / rtc / paint / act_temporal_ensemble / dvac
  chunk_steps: 12
```

单位是模型动作步，非机器人控制 tick。30 Hz 动作的 12 步约为 0.4 秒；
100 Hz 的控制循环不会将其改为 0.12 秒。`policy.horizon_steps` 仍表示模型
预测长度，放瓶子 joint-15000 为 50，采样器的 `num_steps` 也不受此项影响。

该字段统一了配置入口，各算法的实际语义保持不变：

| runtime | 12 的含义 | 兼容的旧字段 |
|---|---|---|
| `manimux` | 原始预测轨迹最多使用前 12 行，延迟裁剪可能减少可用行数 | `max_chunk_steps` |
| `rtc` | 重推理执行窗口下限；结合预测延迟调整，旧 chunk 在推理期间继续执行 | `rtc.min_execute_steps` |
| `paint` | 到第 12 个源动作步后触发下一次推理，等待期间继续执行旧 chunk | `paint.execution_steps` |
| `act_temporal_ensemble` | 每隔 12 个动作步查询新 chunk，多份预测仍参与融合 | `temporal_ensemble.query_interval_steps` |
| `dvac` | 自适应执行长度上限 12，允许算法选择更短的长度 | `dvac.max_execution_steps` |

因此相同数值不保证各算法恰好执行相同数量的动作，更不等于实机已经到达
第 12 行的位置。default 的 refill 设置和各异步算法的延迟约束仍然生效。

配置层会将公共字段转换为算法内部参数；旧配置无需改动。新旧字段同时填写
且数值冲突时直接报错，不静默覆盖。既有边界检查仍生效，例如 PAINT 的初始
延迟不得大于执行窗口，ACT 查询间隔必须小于预测长度。

AAC、AutoHorizon 的长度由各自模型/选择器产生，目前没有等价的固定间隔参数。
给它们配置 `chunk_steps` 会明确报错，避免把自适应算法偷偷改成固定截断。
设置仅在下次加载 runtime 配置时生效，不会自动重启任何服务。

当前 Pi05 和 OpenWAM 放瓶子的串行对照均使用
`inference_schedule: serial`、`chunk_steps: 12`，见
[纯串行执行](serial-execution.md)。Pi05 模型仍预测 50 步，OpenWAM 模型仍预测
32 步；runtime 在 timeline commit 时只保留并播放前 12 步，不按推理延迟裁掉
源轨迹前缀。
