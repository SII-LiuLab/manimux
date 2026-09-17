# ABC + YAM：旧 native 部署已移除

ManiMux 不再提供原生 ABC HTTP server、worker、离线探针和部署配置。
这次是删除旧入口，**不是完成 ABC 的 XPolicyLab 迁移**；当前没有可替代的 ABC 启动命令。

需要恢复 ABC 部署时，请将模型源码、预处理、归一化和 sampler 接入
`XPolicyLab/policy/<POLICY>/`，验证 checkpoint 与 YAM 动作契约后，再添加成对的
server / infra 配置。不要将 SAPolicy 当作 ABC 的等价替代。

历史实现可从 Git 历史查看；已有权重、数据和本地环境未删除。
接入规范见 [AGENTS.md](../AGENTS.md#model-integration-xpolicylab-only)，
当前可用入口见[模型运行手册索引](README.md#policies-and-deployment)。
