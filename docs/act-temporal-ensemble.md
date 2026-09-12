# ACT Temporal Ensembling

ManiMux 的 ACT strategy 复用现有模型服务、PolicyAdapter、ActionTimeline、Executor 和
RobotDriver。它不修改模型权重、输入输出、归一化、关节顺序或 XPolicy server。

## 上游依据

- 官方仓库：<https://github.com/tonyzhaozh/act>
- 固定 revision：`742c753c0d4a5d87076c8f69e5628c79a8cc5488`
- 对应实现：[`imitate_episodes.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L191-L259)

保留的官方核心是：把覆盖同一个绝对动作时刻的预测按产生时间从旧到新排列，再使用

```text
w_i = exp(-0.01 * i) / sum_j exp(-0.01 * j)
```

加权求和。`coefficient: 0.01` 与官方默认一致。

## ManiMux 的调度适配

官方 ACT 在 `temporal_agg` 模式下令 `query_frequency = 1`，即每个 policy action step
查询一次。ManiMux 保持这一行为作为默认值，但增加：

```yaml
execution:
  runtime: act_temporal_ensemble
  blend_steps: 0
  temporal_ensemble:
    coefficient: 0.01
    query_interval_steps: 1
```

`query_interval_steps` 的单位是 policy 轨迹点，不是机器人 control tick。Pi05 的
`action_dt_s` 是约 `33.3 ms`，示例配置使用 `4`，所以目标查询间隔约 `133 ms`。这与
官方默认频率不同，因此准确名称是 **ACT Temporal Ensembling + ManiMux asynchronous
scheduling**，不是未经改动的官方 rollout loop。

选择 `4` 而不是 `3`，是因为此前 Pi05 server 往返通常约 `90–127 ms`：`3` 只有约
`100 ms`，更容易被单次推理占满；`4` 留有少量余量。若模型能够稳定在一个 action step
内返回，可将其改回 `1`，恢复官方查询频率。除此之外不需要修改服务端。

`blend_steps` 必须为 `0`，否则 Timeline 的线性 seam blend 会在 ACT 聚合之后再次改写
轨迹。Smooth/MPC Executor 和安全限制仍照常位于 ACT 之后，它们属于统一真机执行层，
不是 ACT 算法的一部分。

## Pi05 配置入口

模型服务命令与原 ManiMux 实验相同，只切换 infra config：

```bash
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/act-temporal-ensemble.yaml
```

该入口已完成离线公式、配置、Runtime 回归和 Pi05/YAM 真机执行。操作者在 step-1000
红球 checkpoint 上观察到动作连续、整体丝滑；这是一次真机执行与主观连续性证据，不是
任务成功率或统计优势。运行前仍应先按 Pi05 runbook 完成 checkpoint、norm stats、相机和
机器人检查。

本地红球到盒子 step-1000 checkpoint 使用独立的 50-step 配置，不与上面的 16-step
checkpoint 混用：

```bash
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/pick-red-ball-box/act-temporal-ensemble-step1000.yaml
```
