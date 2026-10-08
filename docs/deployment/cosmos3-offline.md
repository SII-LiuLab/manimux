# Cosmos3 离线推理手册

本文只覆盖 **Cosmos3 DROID policy 的离线推理**。不上相机、不启 CAN、不连接 YAM，
也不提供 YAM 配置。公开的
`nvidia/Cosmos3-{Edge,Nano}-Policy-DROID` 是单臂 DROID checkpoint：输入为三张 RGB
图、语言和 `7+1` 维状态，输出为 `32×8` absolute joint-position chunk。

## 实现边界

```text
XPolicy observation
    -> policy/Cosmos3/model.py        只改字段名
    -> official RobolabPolicyService
       model / transforms / normalization / denoise / postprocess 全部官方
    -> official H×8 action
    -> policy/Cosmos3/model.py        只拆成 7 维 arm + 1 维 gripper
```

官方项目入口是 `NVIDIA/cosmos`；当前 DROID 推理实现位于官方
`NVIDIA/cosmos-framework`，在 XPolicy 内作为 nested submodule 固定到
`c7e8d76b5da8aeae38cdac91c6cfd57185b2f6bc`。我们没有复制或修改官方模型源码。

## 1. 初始化源码

```bash
cd /home/ubuntu/manimux
git submodule update --init --recursive
```

确认版本：

```bash
git -C XPolicyLab/policy/Cosmos3/cosmos-framework rev-parse HEAD
```

## 2. 安装官方环境

```bash
cd /home/ubuntu/manimux/XPolicyLab/policy/Cosmos3
bash install.sh
```

默认 Python 是 `/home/ubuntu/manimux/envs/cosmos3/.venv/bin/python`。脚本直接运行官方
`uv sync --all-extras --group cu130-torch213-train --group policy-server`，然后只额外以
editable 方式安装 XPolicyLab，使同一个进程能加载 adapter 和官方 service。

固定版本的 CUDA wheel 使用 CPython 3.13 ABI，因此安装脚本默认 `COSMOS3_PYTHON=3.13`。
本机如果使用 framework 自带的 CUDA 12.8 依赖组，可运行：

```bash
COSMOS3_CUDA_GROUP=cu128-train bash install.sh
```

## 3. 一次离线 forward

这条命令加载官方 Edge DROID checkpoint，用全零 RGB/状态做一次真实模型 forward，只检查
模型能否加载、输出 shape 是否为 `H×8`、数值是否有限：

```bash
cd /home/ubuntu/manimux
envs/cosmos3/.venv/bin/python \
  XPolicyLab/policy/Cosmos3/offline_infer.py \
  --config XPolicyLab/policy/Cosmos3/deploy.yml \
  --prompt "Pick up the object."
```

成功输出示例字段：

```json
{
  "status": "ok",
  "checkpoint": "nvidia/Cosmos3-Edge-Policy-DROID",
  "action_shape": [32, 8],
  "finite": true
}
```

这只证明官方权重经过 XPolicy adapter 完成一次 forward，不证明任何任务成功率。

## 4. XPolicy WebSocket 离线闭环

权重缓存完成后，可运行 XPolicy debug client。它不会进入 simulator 或 real-world client：

```bash
cd /home/ubuntu/manimux/XPolicyLab/policy/Cosmos3
EVAL_ENV_TYPE=debug bash eval.sh \
  Cosmos3 offline edge droid_single joint 0 0 0 uv envs/yam
```

## 配置说明

默认 `deploy.yml` 完整复现 NVIDIA 的 Edge serving recipe：

- `checkpoint_path: nvidia/Cosmos3-Edge-Policy-DROID`；
- `format_prompt_as_json: true`；
- `guidance_interval: [960, 1001]`；
- `num_steps: 4`、`action_chunk_size: 32`、`action_dim: 8`；
- `conditioning_fps: 15.0`、`action_space: joint_pos`。

切换 Nano 时使用 `nvidia/Cosmos3-Nano-Policy-DROID`，并按官方 Nano 命令把
`format_prompt_as_json` 与 `guidance_interval` 设为 `null`。

## 当前验证状态

- ✅ 官方源码 revision、API 和 DROID 输入输出契约已核对；
- ✅ XPolicy adapter 静态检查和数值零改动单测已通过；
- ⚠️ 本轮尚未安装完整 Cosmos 环境、下载 4B 权重或执行 GPU forward；
- ⛔ 未连接相机、CAN、YAM 或其他真机。
