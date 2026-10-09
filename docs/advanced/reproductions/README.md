# Inference Method Reproduction Records

这里保存算法复现的审计记录。它与模型/本体 runbook 分工不同：

- **方法文档**回答“官方做了什么、我们如何接入、方法参数、怎么运行、证据是什么”；
- **模型 runbook**只回答“模型环境、checkpoint、输入输出契约和默认服务怎么启动”。

每个方法在进入真机 gate 前必须有独立文档，并完整记录：

1. 论文、官方仓库、固定 commit 和读取过的源文件；
2. 官方模型、动作空间、horizon、控制频率和评测环境；
3. 逐条公式、默认参数以及代码中的实际执行顺序；
4. ManiMux/XPolicy 的职责边界和全部适配；
5. 与官方一致、等价适配、明确不同、尚未验证的项目；
6. 配置、权重、stats、数据来源及其绑定关系；
7. 离线单测、真实 forward、仿真、真机各自的命令和证据；
8. 已知风险、失败判据、回滚入口和 reviewer checklist。

## Index

| 状态 | 方法 | 方法文档与命令 | 当前 gate |
|---|---|---|---|
| ✅ | Default ManiMux | [`../architecture.md`](../../development/architecture.md) | Current dual-YAM model integrations exercised |
| ✅ | RTC (inference-time Pi-guided) | [`../xpolicylab-runbook.md#rtc-contract`](../../deployment/xpolicylab.md#rtc-contract) | Pi05/YAM hardware complete; measured delay-aware execution |
| ✅ | Adaptive Action Chunking | [`aac.md`](aac.md) · [`aac-pi05.md`](aac-pi05.md) | Pi05 GPU/YAM hardware complete; functional but visibly laggy |
| ✅ | ACT Temporal Ensembling | [`../act-temporal-ensemble.md`](../inference.md) | Pi05/YAM hardware complete; operator observed smooth execution |
| ✅ | PAINT | [`paint-pi05.md`](paint-pi05.md) | Pi05 GPU/YAM hardware complete; operator observed very good continuity |
| ✅ | AutoHorizon | [`autohorizon-pi05.md`](autohorizon-pi05.md) | Pi05/YAM hardware exercised; synchronous inference holds were visibly stop-and-go |

新方法不得只在 README 表格中打勾；表格状态必须能回链到这里的证据。
