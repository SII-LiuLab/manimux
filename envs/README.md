# 硬件/模型环境 —— 请勿用 uv 管理

> **Do not run `uv sync` / `uv run` here.** These are plain venvs created with
> `uv venv`; they hold packages that are not in `uv.lock` (`i2rt`, `torch`,
> `flash-attn`), and any uv project command would uninstall them.

这些目录是用 `uv venv` 建的**普通 venv**，不是 uv 的项目环境。uv 的项目环境只有
仓库根目录的 `.venv` 一个，由 `pyproject.toml` + `uv.lock` 声明。

## 禁止事项

- 不要在这些环境上运行 `uv sync` 或 `uv run`；
- 不要设置 `UV_PROJECT_ENVIRONMENT` 指向它们。

原因：`uv sync` 是**声明式**的 —— 它会把目标环境校准成 `uv.lock` 描述的样子，
**卸掉一切未声明的包**。而这些环境里最关键的依赖恰恰都不在 lock 里
（`i2rt`、各自的 `torch`、`flash-attn` 都是 `uv pip install` 手动装的）。

实测（`--dry-run`，未真正执行）：

```console
$ UV_PROJECT_ENVIRONMENT=envs/yam/.venv uv sync --dev --dry-run
Would uninstall 72 packages
 - i2rt==1.1.2 (from git+https://github.com/i2rt-robotics/i2rt.git@5d47b358...)
 - pyrealsense2==2.58.3.10794
 - torch==2.5.1+cu121
```

`i2rt` 一旦被卸掉，真机 runtime 会在 connect 阶段直接报 `ModuleNotFoundError`。

## 各环境的分工

| 目录 | 跑什么 |
|---|---|
| `../.venv` | uv 托管：开发、mock 运行、Viewer demo；不需要模型依赖 |
| `yam/.venv` | YAM 相机、Viewer、数采和 runtime，含 `i2rt`；不运行模型 |
| `xr1/.venv` | XR-1 XPolicyLab 模型服务；runtime 仍从 `yam` 起 |
| `lingbot-vla2/.venv` | LingBot-VLA2 XPolicyLab 模型服务；runtime 仍从 `yam` 起 |
| `umi_dp/.venv` | UMI DP XPolicyLab 模型/训练；用 `XPolicyLab/policy/UMI_DP/install.sh` 创建 |

模型的 torch / CUDA 版本由对应 XPolicyLab policy 的安装说明管理，不再由 ManiMux
extras 安装。旧 ABC / MolmoAct2 native 入口已移除；本地已有环境不因此自动删除。

## 正确的操作方式

安装或增补依赖时，用 `uv pip install --python` 显式指定解释器，不要用项目命令：

```bash
uv pip install --python envs/yam/.venv/bin/python -e ".[cameras,collection,xpolicylab]"
uv pip install --python envs/yam/.venv/bin/python \
  "git+https://github.com/i2rt-robotics/i2rt.git@5d47b358bafb30c65e397f2ece506550a0db4594"

# XR-1 XPolicy model server dependencies
uv pip install --python envs/xr1/.venv/bin/python -e XPolicyLab
```

运行时一律走显式路径（不要 `source` 之后裸敲命令，容易跑错环境）：

```bash
Y=envs/yam/.venv/bin
$Y/manimux serve --config configs/pi05/yam/infra/put-bottles/rtc-joint-step30000.yaml
```

模型、相机和 Viewer 必须按[统一启动指南](../docs/guideline.md)在各自终端启动。
运行需要已安装 SDK 的离线测试时，显式使用该环境：

```bash
envs/yam/.venv/bin/pytest tests/unit tests/integration
```

模型环境步骤见 [Pi05](../docs/pi05-yam-runbook.md)、
[XR-1](../docs/xiaomi-xr1-yam-runbook.md) 和 [XPolicyLab](../docs/xpolicylab-runbook.md)。
