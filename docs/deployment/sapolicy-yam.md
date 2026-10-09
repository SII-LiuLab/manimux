# SAPolicy + YAM

## 当前部署

MV51 RAW 通过 `XPolicyLab/policy/SAPolicy` 和共享 `xpolicylab_ws` 接入。
模型源码、依赖安装、数据与训练入口均在该 policy 目录中；ManiMux 负责相机、
观测映射、YAM FK/IK、调度、执行与记录。完整安装和资源准备见
[SAPolicy README](../../XPolicyLab/policy/SAPolicy/README.md)。

支持 `yam_dual / ee`。模型返回标准动作字典：双臂绝对末端位姿采用
`[x,y,z,qw,qx,qy,qz]`，夹爪连续开度 0 闭合、1 张开。YAM 适配器转换为
两组各 7 维的关节目标。旧 `packed_ee_wire` 的 xyzw 格式仍可显式使用，
与标准接口共用同一推理和执行语义。

| 参数 | MV51 RAW 设置 |
| --- | --- |
| checkpoint | `ft_teleopMV51_from_abc_notcpkv_rot5_bs1024_raw.ckpt` |
| 权重 | RAW，`use_ema: false` |
| 观测历史 | 1 帧，外部视角 + 左右腕 |
| action horizon / sampler steps | 50 / 10 |
| 普通 chunk 执行前缀 | 前 25 个源时间步中仍有效的部分 |
| RTC | horizon 50；执行下限 25；初始 delay 4；延迟窗口 10；beta 5 |
| smoother | 共享 `SmoothExecutor`，`tracking_mode: legacy`，8 Hz 低通 |
| 模型动作间隔 / 控制频率 | 1/30 s / 100 Hz |
| 关节速度 / 加速度限制 | 均关闭（显式 `null`） |
| 抓取 / 释放等待状态机 | 当前低通与 RTC 配置均不启用 |
| 夹爪速度 / 加速度 / 单独闭合限速 | 均关闭（显式 `null`） |
| TCP 参考点 | 训练 MJCF 的 grasp_site，额外 offset = 0 |
| 图像 | RGB；640×480 → 224×168，模型中心裁剪 210×154 |

同一执行模式下的三个 infra 配置只改变外部相机、视角标识和记录目录。模型文件 SHA-256 在
启动时核验，runtime 用该摘要匹配后端身份；不再把固定机器的绝对权重路径当作身份。
旧 teleop50 / ABC 配置保留为历史兼容路径，不作为 MV51 的启动入口。

MV51 普通和 RTC 共六个配置显式关闭软件速率限制；直接删除这些字段会恢复默认限速。
位置边界独立保留。低通 smoother 使用连续夹爪，不启用原 braking 的抓取/释放等待。
共享 YAM 配置仍包含
夹爪单独闭合限速，因此这里明确使用 MV51 的完整无速率限制设置。
修改后须重新启动 `manimux serve` 才会加载新值。

## 安装与模型服务

从 ManiMux 根目录执行。先按照 policy README 用原始权重、DINOv2 权重和
匹配 normalizer 创建 `checkpoints/finetuned/sapolicy/teleopMV51/` 资源包。
模型权重和本机记录不进入 Git。

```bash
bash XPolicyLab/policy/SAPolicy/install.sh
XPolicyLab/policy/SAPolicy/.venv/bin/python manimux/servers/sapolicy.py \
  --config manimux/configs/policy/sapolicy/yam/teleopMV51/raw.yaml --check
XPolicyLab/policy/SAPolicy/.venv/bin/python manimux/servers/sapolicy.py \
  --config manimux/configs/policy/sapolicy/yam/teleopMV51/raw.yaml
```

`--check` 只验证配置和资源。共享模型服务默认监听 `ws://127.0.0.1:8510`，
不连接真机。服务内部使用 policy 目录中的 resolved 配方，不依赖外部源码或
某次实验生成的 YAML。

## 相机与 GUI

已有相机服务时直接复用；以下命令用于服务尚未启动的情况。

```bash
envs/yam/.venv/bin/manimux-camera-server --config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml
envs/yam/.venv/bin/manimux-camera-server \
  --config manimux/configs/embodiment/sensor/cameras/gemini_2_views.yaml \
  --rep-endpoint tcp://127.0.0.1:5575 --pub-endpoint tcp://127.0.0.1:5576
envs/yam/.venv/bin/manimux-robogui --robot yam --host 0.0.0.0 --port 8086
```

| 视角配置 | 外部相机 | 腕部相机 |
| --- | --- | --- |
| `top.yaml` | `d405_front`，RealSense Top | `d405_left`、`d405_right` |
| `gemini305.yaml` | Gemini305，序列号 `CV278640000Z` | 同上 |
| `gemini335.yaml` | Gemini335，序列号 `CP0N763000LK` | 同上 |

Gemini 按设备序列号定位 UVC RGB 节点。每轮只请求和检查选中的三路相机。
GUI 默认 `camera_mode: policy`，跟随 runtime 上报的 `policy.adapter.camera_map`，
主画面标注模型输入名，左右小画面使用简洁的 `left side` / `right side` 标签；
模型输入与物理相机的完整对应关系显示在右侧表格中。
模型配置声明输入视角；部署配置绑定本机相机；RoboGUI 自动展示，不需要重复绑定。
输入按映射顺序排列，第一个是主画面，Top 叠加预览跟随主画面。
超过三路的输入显示在“更多输入相机”区域。
未收到模型输入配置时显示默认 `top / left / right` 预览，并注明尚未获取配置。
新一轮、新 runtime、失联和视角变化会清理旧画面；缺失的选中视角显示“等待图像”。

调试时仍可用 `--config manimux/configs/robogui/yam-top.yaml`、`yam-gemini305.yaml` 或
`yam-gemini335.yaml`（后两者同在 `manimux/configs/robogui/`）进入 `camera_mode: manual`。
这只覆盖预览；界面同时列出模型输入，预览选择不改变送入模型的相机。
若要恢复自动跟随，省略 RoboGUI 的 `--config` 或显式设置 `camera_mode: policy`。
记录仍保存物理相机名和 `view_profile`。

## Rollout

```bash
envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_mv51_top.yaml
# 使用 RTC：关闭上一 runtime 后选择此配置
envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_mv51_top_rtc.yaml
```

打开 `http://localhost:8086`，点击 **Prepare normal rollout**，摆好场景后
点击 **Start rollout**。**Pause / Hold** 暂停；**Finish & Home** 保存本轮并
按当前配置让双臂同时 Home（5 秒）。切换视角前结束本轮并关闭该 runtime，
再用 `yam_sapolicy_mv51_gemini305.yaml` 或 `yam_sapolicy_mv51_gemini335.yaml` 启动下一轮。默认模式下 GUI 自动跟随，
无需随视角重启；相机与模型服务可复用。
RTC 对应 `yam_sapolicy_mv51_gemini305_rtc.yaml` 和 `yam_sapolicy_mv51_gemini335_rtc.yaml`；使用同一模型服务。
后端实际实现并声明 `rtc` 能力后 runtime 才能启动，身份检查保持启用。

记录位于 `data/sapolicy/teleopMV51/<view>/session-*/rollout-*`，RTC 的 view 后缀为 `-rtc`。
`result.success` 只表示运行流程完成；投瓶是否成功由实际观察或人工评价确定。
当前执行采用进程 IK、双臂成对提交和低通平滑。原有双臂独立提交依赖 braking，
因此不用于这两套配置。RTC 对旧关节轨迹先做同坐标系 FK，再在 SA 中转换为相对
动作并归一化，由实际 DiT 采样器做 VJP 引导。普通 chunk 在交接处 blend 4 步；
RTC 首段也 blend 4 步，已有条件的后续 chunk 不叠加 blend。
RTC 的 25 步是调度下限，不是固定裁掉后半段；根据实测推理和解码延迟调整。
历史 braking 记录不能当作新 smoother/RTC 的真机验证。

## tcp_rot20：ABC bottles 零样本

`tcp_rot20` 是训练方提供的双臂 3 相机 TCP put-bottles 检查点（RAW，step 50k），
未用本站数据微调。

### 资源包

`checkpoints/finetuned/sapolicy/tcp_rot20/` 不进入 Git，在机器之间用 `rsync -aL` 复制
（`-L` 解引用本机的软链接）：

| 文件 | 内容 |
| --- | --- |
| `model.ckpt` | 训练检查点，`sha256:d84e94f2…b032825a3f`（配置中有完整值，服务启动时校验） |
| `backbone.pth` | DINOv2 初始化权重（与 teleopMV51 相同），随后被检查点严格加载覆盖 |
| `action_normalizer.pt` | 与检查点匹配的动作归一化 |
| `resolved_config.yaml` | 训练时展开的配置 |
| `resolved_config.compiled.yaml` | 同上，另开启 `compile_backbone` / `compile_action_head` |
| `source/` | SpatialAlignVLA rollsweep0917 源码快照，加了推理时 RTC 补丁；普通采样路径与原版逐位一致 |

```bash
rsync -aL --info=progress2 <本机>:~/manimux/checkpoints/finetuned/sapolicy/tcp_rot20/ \
  checkpoints/finetuned/sapolicy/tcp_rot20/
```

### 启动

```bash
XPolicyLab/policy/SAPolicy/.venv/bin/python manimux/servers/sapolicy.py \
  --config manimux/configs/policy/sapolicy/yam/tcp_rot20/raw-50k.yaml
envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_tcp_rot20_async40_cranksite.yaml
```

服务端推理约 60–90 ms（4090，未编译）。如需更快，把 `cfg_file` 换成
`resolved_config.compiled.yaml` 并设置 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`（约 36 ms）；
前几次请求会编译数分钟，开始 rollout 前先发请求预热。

### 设置及原因

| 项 | 设置 | 原因 |
| --- | --- | --- |
| 调度 | ManiMux async，每块执行 40/50 步，blend 4，refill 0.2 s | obs_hist=1，模型每块都把下降放在后半段；只执行 16–25 步会一直悬停 |
| IK 解码 | `action_decoding: process` | 控制线程内 IK 每次交接卡顿 30–100 ms，表现为手臂摇晃 |
| 执行器 | smooth braking，0.6 rad/s，1.5 rad/s² | 沿用 `yam_smooth_braking.yaml` 的实测值；高频抖动 2.4 → 1.05 °/s |
| 夹爪 | continuous，不限速，无守护 | — |
| 起始位 | ABC-130k bottles ready 姿态中位数，4 s | ABC 数据没有从零位开始的 episode |
| 安装高度 | `model_from_kinematics` z +2.2 cm | 本站底座垫板 2.2 cm，已用 J2 轴高和零位指尖高实测确认 |
| TCP 参考点 | `tcp_rot20/crank_site_assets`：linear_4310 grasp_site 沿 site x 移 −4.1 cm | ABC 约 59% 使用 crank_4310，指尖相对 grasp_site 偏 4.1 cm；只改一个 site，网格软链接到默认资源 |

### 已知情况

- test-time RTC 无效：去噪器对输入的敏感度 ∂x̂/∂x 只有约 0.01–0.06。
- 同一观测重复请求，在 ABC ready 起始位附近前 16 行相差 0.5–3.5 cm，块末端相差 3–10 cm。
  后端没有 `inference_seed`。
- 抓取闭合时指尖高度基本不随 z 偏移或 crank 修正变化，最后接近由视觉驱动。
  剩下的偏差来自模型对本站画面的判断，不是安装补偿能解决的。

## 离线验证

### 使用其他 SAPolicy 训练检查点

为检查点单独配置 `model_path`、训练时展开的 `cfg_file`、匹配的
`normalizer_path`、`backbone_path` 和 `checkpoint_source`。需要匹配训练源码时，
用 `sapolicy_root` 指向该快照；适配器保留当前归一化和 TCP 位姿转换包装器，
只从快照导入模型核心。切换源码快照需要新的模型进程。

TCP + action 联合训练的 `dataset_opts` 可以是嵌套列表。适配器按
`combined_loader_opts` 选择最后的动作分支，检查其中所有数据集的动作语义、
图像预处理和相机顺序一致，再用于推理。权重使用严格加载，缺失或多余参数会报错。
旧训练快照未实现 RTC 时只声明普通采样能力，不能直接配用 MV51 RTC 实验。

以下命令仍以 MV51 为例；新检查点应换成自身配置与权重路径。

```bash
envs/yam/.venv/bin/python -m pytest -o addopts='' -q \
  XPolicyLab/policy/SAPolicy/tests tests/unit/test_sapolicy_xpl_model.py \
  tests/unit/test_sapolicy_xpl_transport.py tests/unit/test_sapolicy_decode_timing.py
XPolicyLab/policy/SAPolicy/.venv/bin/python -m pytest -o addopts='' -q \
  XPolicyLab/policy/SAPolicy/tests
XPolicyLab/policy/SAPolicy/.venv/bin/python -m XPolicyLab.policy.SAPolicy.validate_checkpoint \
  --model-config manimux/configs/policy/sapolicy/yam/teleopMV51/raw.yaml \
  --checkpoint "$PWD/checkpoints/finetuned/sapolicy/teleopMV51" --rtc \
  --output data/sapolicy/checkpoint-validation.json
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config manimux/configs/experiments/put_bottles/sapolicy/yam_sapolicy_mv51_top_rtc.yaml
```

最后一条需要模型服务，只发送合成图像和配置中的关节状态，不打开相机或控制
机器人。标准 debug、编码图像、batch、原生数据处理与训练入口见 policy README。

2026-09-13 验证覆盖全新环境安装、真实 GPU forward、标准动作/旧格式数值
一致性、共享服务器、YAM 离线 IK、batch 与 reset、原生 ABC 读取以及合成数据上
一次 GPU 训练步骤和 checkpoint 保存。标准接口迁移后已进行一次 Top 真机 rollout，
由操作者结束并保存；该轮仍使用旧限速，未标注任务成功率。

随后完成 smoother / RTC 离线验证：真实权重三个固定种子的重叠区误差均下降，
零权重 RTC 与普通采样逐值一致，reset 和条件清理通过；共享 debug 普通图像、
编码图像 batch，以及三个视角的 RTC → YAM IK 均通过。合成观测下 RTC forward
约 172–183 ms，热请求加双臂 IK 约 310 ms。新 smoother / RTC 尚未进行真机 rollout。
