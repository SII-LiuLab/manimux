# 服务入口

从仓库根目录启动；解释器必须来自对应的硬件或模型环境。

| 模块 | 职责 |
| --- | --- |
| `manimux.servers.camera.server` | 打开配置的相机，提供 ZMQ 图像服务 |
| `manimux.servers.pi05` | 读取 Pi05 配置，启动 XPolicyLab 服务 |
| `manimux.servers.groot` / `xr1` / `sapolicy` / `isaac05` | 各模型已有的 XPolicyLab 启动与配置入口 |
| `manimux.servers.openwam` / `umi_dp` | 启动 XPolicyLab，也支持已有的 checkpoint 配置绑定流程 |

模型网络、权重加载、归一化和采样代码仍在 `XPolicyLab/policy/`；这里没有另一套模型实现。
相机客户端和 runtime 传感器位于 `manimux/embodiments/sensor/camera_server/`，
读取网络图像不会再创建物理相机。

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --config manimux/configs/embodiment/sensor/cameras/yam.yaml

XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --config manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml
```

模型入口使用仓库中的 XPolicyLab 子模块及各自环境；搬动启动文件不会安装模型、
下载 checkpoint 或初始化子模块。`--help` 只展示参数，不启动服务。
