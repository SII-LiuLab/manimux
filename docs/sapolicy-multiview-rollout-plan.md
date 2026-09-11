# SAPolicy MV51 真机视角切换方案

2026-09-11；已核对采集对话、当前代码及 QZ 训练配置。本文件是修改方案，尚未实现或进行真机验证。

模型每次使用两个 wrist 和一个外部视角。建议提供 `top`、`gemini305`、`gemini335` 三个 rollout 预设，由同一份执行配置生成；在每个 episode 开始前选择，并在该 episode 内固定。

## 已确认的训练约定

用户指定的配置是 `exp_dinov2lfrz_yam_bottles_ft_teleopMV51_from_abc_notcpkv_rot5_dit_8gpu_bs1024.yaml`。QZ 输出目录内的 `resolved_config.yaml` 包含：

```yaml
camera_pair_choices:
  canonical_names: [top, left, right]
  choices:
    - [top, left, right]
    - [gemini305, left, right]
    - [gemini335, left, right]
  enumerate_all: true
```

`num_cameras=3`、`obs_hist_length=1`、`use_depth=false`、`use_state=true`、动作 horizon 50、10 次推理步；RAW 权重使用 `use_ema=false`。腕部通过 `aux_bypass_cameras: [left, right]` 绕过辅助头，因此服务端始终保留模型名称 `top/left/right`，物理 Gemini 也映射到 `top`。把三个外部相机同时送入会改变这份 checkpoint 的输入约定。

采集配置与原对话确认了五路 RGB，均为 640×480、30 FPS：

| 物理相机 | 序列号 | ManiMux 建议沿用/新增的 sensor key |
| --- | --- | --- |
| left wrist | 260322276964 | left_camera |
| right wrist | 260322274672 | right_camera |
| top RealSense | 260322276687 | front_camera |
| Gemini 305 | CV278640000Z | gemini305 |
| Gemini 335 | CP0N763000LK | gemini335 |

序列号来自当前采集配置，未在本次工作中打开相机验证在线状态。

## 视角预设

| 选择 | 模型 top 来源 | 模型 left 来源 | 模型 right 来源 |
| --- | --- | --- | --- |
| top | front_camera | left_camera | right_camera |
| gemini305 | gemini305 | left_camera | right_camera |
| gemini335 | gemini335 | left_camera | right_camera |

例如 Gemini 305 预设最终解析为已有配置结构：

```yaml
sensors:
  - name: yam_cameras
    driver: camera_server
    options:
      camera_names: [gemini305, left_camera, right_camera]
policy:
  options:
    camera_map:
      top: gemini305
      left: left_camera
      right: right_camera
```

以上只是需要变化的字段，不能作为完整运行配置直接启动。`camera_map` 已同时供 SAPolicy adapter 和 XPolicyLab observation encoder 使用，无需增加模型视角名称或修改动作协议。视角选择器应一次性解析 sensor 列表、camera_map、相机参数和记录信息，避免手动改动多份 YAML 后出现错配。

## 需要修改的位置

1. **统一相机服务支持 Orbbec。** `src/manimux/sensors/camera_server/server.py` 的构建函数目前只创建 RealSense。增加 camera type 工厂及 Orbbec RGB 驱动，复用 `yam-abc-reproduce/yam_abc_reproduce/camera/orbbec.py` 的发现逻辑：USB vendor `2bc5`、Gemini product `0840/0800`、interface `04`、capture capability、serial。使用 MJPG，正确转换 BGR→RGB，保持 640×480 以及原始方向。将这段驱动能力放入 ManiMux，避免运行时依赖另一 checkout。

2. **按本次选择读取、检查三路相机。** 服务端当前 `_snapshot()` 读取全部设备，client 又先检查全部时间戳，sensor driver 最后才筛选 `camera_names`。若直接启动五路，未使用的 Gemini 掉线也可能中断 top 测试。给 `obs` 请求增加可选 `camera_names`，服务端按请求读取，响应按相机报告健康状态；客户端仅要求所选三路存在、时间戳有效、帧未过期。未指定名称的旧客户端保留原协议行为。可让相机服务长期持有五个 worker，设备异常按相机隔离；选中异常设备时明确报错，不自动替换视角。预览帧时间必须来自最近一次成功采集，不能把读取缓存的时间当采集时间。

3. **用一个入口选择 profile。** 将相机清单和三个预设独立于 SAPolicy 执行参数保存，先支持启动参数/配置中的 `view_profile`。复用当前 `manimux-braking-h25.yaml` 的执行设置，三个测试只改变观测视角；所有预设指向 MV51 RAW checkpoint 和匹配配置。仍保留 `expected_backend.model.model_path` 校验，防止误连 teleop50 服务。服务端可以保持加载同一个 MV51 模型。

4. **GUI 显示并锁定实际视角。** `src/manimux/viewer/dashboard.py` 当前固定三个图像槽和 top 标签；`viewer/robots/yam.py` 不认识 Gemini 的显示映射。新增外部视角选择，在 Prepare 前生效，当前 episode 内锁定；显示 `Gemini 305 + left wrist + right wrist`。预览布局仍可保留三个槽，但标签和映射由实际配置提供。下一 episode 重新解析选择并清理旧请求、缓存和动作队列。

5. **保存可比较的实验记录。** `src/manimux/runtime/edge.py` 已向 recorder 写入 checkpoint 后端等信息，但未专门保存视角映射。给 episode metadata 增加 `view_profile`、`camera_map`、serial、分辨率、预处理版本、内参模式、checkpoint SHA-256 和解析后的配置哈希。视频保留物理相机名称，结果按视角汇总；未送入模型的监控视频另行标记。

## 内参处理

目前训练 loader 有一个需要明确记录的兼容行为：`abc_episode.py:1204` 从 episode 的 top TCP 标签取 `K224`，并把同一矩阵写入所有 canonical camera。Gemini 替换 top 时，没有切换到 Gemini 的真实标定；其 `tcp_valid` 被置零。采集 metadata 也没有保存 Gemini 的标定矩阵。

建议把物理相机标定和模型实际接收的 K 分开记录。此次基线可显式使用与训练一致的 reference K，并标注其来源；后续标定模式使用 serial 对应的真实 K，经 resize/crop 一致变换。不要把 reference K 标作 Gemini 的实测内参。

进一步检查模型代码可见：本配方 `use_depth=false` 使 LatentTrunk 与 DiT 的几何注入关闭，`disable_tcp_kv=true` 关闭 TCP 条件进入动作头的路径。因此从代码路径判断，当前动作推理不依赖 K 的数值；这个判断尚未用该权重做固定随机噪声的数值对比。传输协议仍要求每路 K，保留兼容字段即可，不需要为了此次 RGB 视角实验先引入深度或外参依赖。若以后启用几何/TCP 条件，应重新核对训练和推理的标定约定。

## 验证与实施顺序

先实现 Orbbec 驱动、按选定相机请求和三个 profile，再接 GUI 选择和记录字段。必要的离线检查包括：

- 用相同一条五路录制 episode 和同一时刻的 state，分别生成三个请求；断言仅 top 图像来源变化，两 wrist 相同，模型名称始终是 top/left/right。
- 比较源图像和模型收到的图像，验证 RGB、方向、640×480→224×168 resize 及评估 crop 一致；模型输出有限且满足现有动作解码约定。
- 验证选中相机缺失/过期时明确失败，未选相机异常不阻断当前三路；检查 capture timestamp 没有被缓存读取时间刷新。
- 固定采样噪声比较 K 模式，确认该 checkpoint 的数值行为，再决定是否把真实 K 模式作为独立实验。
- 检查 GUI 实际显示和 episode metadata 的视角一致，无法在运行中的 episode 混入另一个 profile。

之后分别进行三组真机 rollout，复用同一模型、动作执行参数和场景布置，并记录每组任务结果。离线请求通过和相机预览可用各自单独报告，不视为任务成功。

## 本次证据文件

`/home/ubuntu/sa/deployments/teleopMV51_qz/` 保存了用户指定训练 YAML、实际 resolved_config、数据组件 YAML、两个 normalizer 文件及用于检查的代码片段。`SHA256SUMS` 和 `provenance.json` 对应远端文件的独立哈希与路径。

其中 `source/` 是本次从 QZ `SpatialAlignVLA_mvft` 读取的检查材料，不是完整可运行部署；未证明这份可变 checkout 的所有代码都与训练时完全一致。对应配置中的视角选择已在训练输出目录保存的 resolved_config 中独立确认。
