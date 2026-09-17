# MolmoAct2 + YAM：旧 native 部署已移除

ManiMux 不再提供原生 MolmoAct2 HTTP server、worker、独立 Viewer 启动器和部署配置。
模型源码保留在 [XPolicyLab/policy/MolmoACT2](../XPolicyLab/policy/MolmoACT2/)；
**模型目录存在不代表旧 YAM checkpoint 的部署已完成迁移或验证**。

重新提供 YAM 部署前，需要确认对应 checkpoint 的相机、状态、动作和归一化契约，
通过 XPolicyLab adapter 与共享服务验证，再添加成对的 server / infra 配置和运行命令。
不要恢复 ManiMux 内的第二份模型实现。

历史实现可从 Git 历史查看；已有权重、数据和本地环境未删除。
接入规范见 [AGENTS.md](../AGENTS.md#model-integration-xpolicylab-only)，
相机、Viewer 和 runtime 启动流程见[统一启动指南](guideline.md)。
