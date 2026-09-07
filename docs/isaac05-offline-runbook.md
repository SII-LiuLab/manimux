# Isaac 0.5 离线接入手册

本接入使用 [PerceptronAI 官方 Isaac 仓库](https://github.com/perceptron-ai-inc/isaac)
和 [Isaac-0.5 官方权重](https://huggingface.co/PerceptronAI/Isaac-0.5)。模型内部继续运行
官方 `PerceptronIsaacPolicy`、官方 observation preparation、preprocessor、Flow expert、
`predict_action_chunk` 和 postprocessor；XPolicyLab 只负责标准 WebSocket 契约。

## 当前边界

公开 checkpoint 是 **LIBERO Spatial** policy，不是 YAM policy：

```text
2 x RGB (image, wrist_image) + 8D LIBERO state + language
  -> official Isaac 0.5 Flow policy
  -> 50 x 7 checkpoint-native absolute EE actions
  -> expose 8 x 7 actions per synchronous XPolicy request at 20 Hz
```

不要把这组 `7D` 动作直接发送给 YAM。YAM 需要单独微调的 checkpoint、匹配的 stats，
以及经过验证的 Isaac-to-YAM action adapter。在这些条件满足前，本接入只做模型服务和
离线 forward 验证，不提供 `manimux run/serve` 真机配置。

## 固定版本

- Isaac repository: `be6507b4aed7472f2029606c22684d4ebc9d73e6`
- Perceptron LeRobot: `e12389c1f8f591ad05dced4e284d4e92e48c5df4`
- XPolicy source: `XPolicyLab/policy/Isaac_05/lerobot`
- Server config: `configs/isaac05/libero/server/base.yaml`

模型卡要求通过 Perceptron 的 LeRobot fork 使用该 checkpoint；stock Transformers 或
stock LeRobot 不是兼容入口。当前官方 policy 也明确拒绝 LeRobot RTC，因此本接入只声明
`default` sampling，不伪造 RTC 支持。

## 1. 下载权重

完整仓库约 `143 GB`。`hf download` 支持断点续传：

```bash
cd /home/ubuntu/manimux
mkdir -p checkpoints/pretrained/perceptron-ai/isaac-0.5
hf download PerceptronAI/Isaac-0.5 \
  --local-dir checkpoints/pretrained/perceptron-ai/isaac-0.5
```

当前机器已使用 `tmux` 会话 `isaac05-download` 挂起下载。查看进度：

```bash
tmux attach -t isaac05-download
tail -f data/download-logs/isaac-0.5.log
du -sh checkpoints/pretrained/perceptron-ai/isaac-0.5
```

不要同时启动第二个下载进程。

## 2. 初始化源码与环境

克隆父仓库后先取回 XPolicyLab 及其 Isaac 源码：

```bash
git submodule update --init XPolicyLab
git -C XPolicyLab submodule update --init policy/Isaac_05/lerobot
bash XPolicyLab/policy/Isaac_05/install.sh
```

安装脚本使用官方 lockfile 和 CUDA extra。官方源码同时要求 production
`trained_policy` 使用 PyTorch `2.10.0+cu128`、Transformers `5.5.4`、
`torch_sdpa_v1` 和 NVIDIA H100；当前 lockfile 默认的 PyTorch `2.11` 仅能安装依赖和
运行不分配完整模型的检查。若官方环境不能导入或渲染，不要用相似实现替换模型内部逻辑。

## 3. 静态契约检查

该命令不加载 Torch 权重、不占用 GPU、不启动端口：

```bash
envs/yam/.venv/bin/python scripts/servers/isaac05_server.py --check \
  --config configs/isaac05/libero/server/base.yaml
```

下载完成前会返回 `status: blocked` 和非零退出码。只有 source revision、配置、adapter、
stats、权重索引以及索引引用的全部 shard 都存在时才返回 `status: ready`。

## 4. 模型服务与离线 forward

该模型约 `36B` 参数，公开仓库约 `143 GB`；官方代码还会拒绝非 H100 的 production
CUDA runtime，因此当前 RTX 4090 不能执行这份公开 checkpoint 的官方 forward。以下
命令只应在满足上述官方 ABI 和硬件要求的机器上由操作者运行。

Terminal 1：

```bash
bash XPolicyLab/policy/Isaac_05/setup_eval_policy_server.sh \
  configs/isaac05/libero/server/base.yaml
```

Terminal 2：

```bash
envs/isaac-0.5/.venv/bin/python scripts/validation/isaac05_forward_probe.py \
  --server ws://127.0.0.1:8504 \
  --instruction "Pick up the object."
```

验收输出必须是有限的 `8 x 7` action chunk。这个 probe 使用合成的 LIBERO-shaped 输入，
只证明官方模型链和 XPolicy wire 可以完成一次 forward，不证明仿真成功，更不证明 YAM
真机能力。

## 已验证证据

- 官方源码 revision、checkpoint manifest、deployment adapter 和 native stats 能静态对齐。
- ManiMux wire codec 的 `2 camera + 8D state -> 8 x 7 action` 单元测试通过。
- XPolicy observation/action adapter 单元测试通过。
- 当前未完成完整权重下载、GPU model load、真实 forward、仿真或真机验证。
