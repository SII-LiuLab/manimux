# ManiMux 代码组织

本文件区分目标布局与当前迁移状态。Tianji–TacCap 使用分体装配，YAM 已有臂爪一体
组件；YAM 实验与采集已切换到同一装配入口，RealSense 已统一为 sensor 组件。XPolicyLab 是独立 Git 仓库。

## 当前主要布局

```text
repository/
├── README.md / pyproject.toml / uv.lock
├── manimux/
│   ├── configs/
│   │   ├── experiments/<task>/ # 实验入口；关键选择和动作时间直接可见
│   │   ├── embodiment/        # 组件、整机、IK 与相机组合
│   │   ├── policy/<model>/    # Checkpoint and inference deployment recipes
│   │   ├── inference/         # 推理算法参数
│   │   ├── executor/          # 平滑与执行限制
│   │   ├── collection/        # 数采入口
│   │   ├── viewer/            # 显示配置
│   │   └── local/             # 工位模板
│   ├── embodiments/
│   │   ├── arm/               # 手臂、运动学、组件资源
│   │   ├── end_effector/      # 末端、工具几何、组件资源
│   │   ├── sensor/            # 设备、相机客户端与网络传感器
│   │   └── robot/             # 整机组件管理、RobotModel
│   ├── servers/               # 相机服务与 XPolicyLab 模型启动入口
│   ├── policy_adapter/        # 观测/动作转换及必要 FK/IK
│   ├── policies/              # worker、解码进程、模型客户端
│   ├── kinematics/            # 公共运动学接口与组合算法
│   ├── runtime/               # 调度、时间线、执行器和必要保护
│   ├── collection/ / viewer/ / recording/ / evaluation/
│   └── cli.py / session.py / types.py / clock.py / __main__.py
├── XPolicyLab/                # 独立仓库：学习模型实现与模型服务
├── env_cfg/                   # XPolicyLab 当前固定读取的外部机器人维度接口
├── tests/ / docs/ / scripts/  # 测试、文档、安装与维护工具
├── .local/                    # 本机绑定、集群任务；不提交
└── data/                      # 运行输出；不提交
```

这个布局不要求单独的 Python 配置模块，也不要求为每份 YAML 定义配置类。
`read_yaml()` 只把 YAML 读成字典，与业务字段无关。实验引用、相对路径和 local
合并遵循明确规则，由入口组织；各组件解释自己的参数并完成构造。
不自动加载任意路径字符串，读取配置也不连接硬件。

注册功能放在所属模块：根据配置中的名称选择对应实现。目标不依赖第三方包的
entry point 自动发现；现有动态导入配置须先迁移，再移除通用插件加载器。

`src/` 已移除，Python 模块名仍为 `manimux`。代码和 YAML 以相同相对布局打包，
Viewer 无需区分源码和 wheel 的资源路径。实验动作间隔由 `policy.action_dt_s`
显式声明，下发频率由 `robot.control_hz` 声明；本体 profile 不决定模型动作时间。

根目录的 `env_cfg/` 是当前 XPolicyLab `utils/process_data.py` 从其父目录固定读取的
机器人维度文件，模型打包动作仍依赖它。它不参与 ManiMux 实验配置合并；在上游支持
显式资源位置之前保留原位。`assets/` 是 README 展示素材，`envs/` 是环境说明，
`licenses/` 保存第三方许可证；这些也不属于实验配置。

服务入口集中在 `manimux/servers/`，具体实现按职责放置。学习模型和模型服务仍在 XPolicyLab，
ManiMux 中的模型服务脚本只负责读取配置和启动它。相机服务持有物理相机时，runtime
读取网络数据源，不重复打开同一设备。

## 当前迁移状态

- `cli.py` 已提供 `read_yaml()`、`read_experiment()`、`load_local()`，直接返回字典。
  相机服务和 Tianji 策略服务入口复用这些函数。
- `config_files.py` 及其中的 `RobotBindings`、`LocalBindings` 配置类已移除。
  相机流与整机组件的映射放回相机服务。
- 顶层 `config.py` 已移除。runtime、机器人注册、策略、工作进程、采集后端及脚本
  使用普通字典。完整实验入口是 `cli.load_config()`：读取与引用解析完成后，调用
  各模块的普通参数函数补齐原默认值。`prepare_experiment()` 不构造硬件对象。
- 模块的参数函数放在对应实现旁边，未新增独立的配置类或配置文件框架。运动限位、
  调度组合及后端身份约束继续保留；不再通过 Pydantic 对每个业务字段逐项检查。
  字典没有“哪些字段由用户显式赋值”的隐藏状态，重用已补齐默认值的参数时，
  与默认值相同的无关调度字段不会被当成新的模式覆盖。
- 配置字典直接传给推理和动作解码子进程。动作解码进程仍从整机配置独立创建
  离线运动学，不传递已连接的 robot 对象。
- 实例化链路仍是 `cli → build_runtime → EdgeRuntime → build_robot → from_config`。
  独立相机服务持有物理设备时，runtime 只启动网络传感器；整机传感器保持未启动。
- 这里移除的是原顶层配置类及其调用；YAM 采集 GUI 的工位数据类、Viewer 面板
  配置及官方运动学参数不属于这个顶层实验配置接口，未做无关重写。
- 顶层 `robots/` 已整目录删除。runtime 和工厂直接使用 `embodiments.robot.RobotBase`，
  整机负责提供统一控制方法与运动学，不再保留重复的机器人 Protocol 或旧导入转发。
- 顶层 `end_effectors/`、`sensors/` 已删除，内部调用直接导入 `embodiments` 实现，
  不保留旧路径转发。Orbbec 相机实现位于 `embodiments/sensor/orbbec/`。
- 传感器构造与默认参数位于 `embodiments/sensor/__init__.py`。统一的 `SensorBase`
  支持单帧和命名帧集合；网络数据源由 `embodiments/sensor/camera_server/` 实现。
  runtime 与整机读取均保留原帧对象、时间戳和序号；测试假相机位于 `tests/support/`。
- 实际 adapter 已迁入 `policy_adapter/`，XPolicyLab 客户端和 wire codec 位于
  `policies/xpolicylab/`。旧纯 adapter 包和旧方法签名兼容分支已移除。
- 实验入口归 `manimux/configs/experiments/<task>/`，server 配置归 `manimux/configs/policy/<model>/`。
  算法使用 `inference.algorithm`，执行器使用 `executor.type`，adapter 使用
  `policy.adapter.type`；adapter 参数直接写在实验中。
- `integrations/` 仍有旧 native 模型源码与工具；HTTP 客户端已归入 `policies/`。此次 adapter 迁移未宣称这些
  学习模型已迁入 XPolicyLab。`plugins.py` 与公共 `kinematics/` 保持各自职责。
- Tianji 和 YAM Viewer 均通过 `RobotView` 使用 `RobotModel`，显示配置位于
  `viewer/robots/{tianji,yam}/viewer.yaml`；消息直接保留组名。旧 YAM 显示适配仍供历史调用使用。
  入口与配置见 [Viewer](viewer.md)。

## 共用实现的位置

- YAML 整机装配只有 `RobotModel.from_config()`；原先仅供测试使用的
  `build_tianji_taccap_kinematics()` 与硬编码安装矩阵已删除。测试同样读取装配 YAML。
- TacCap 默认 TCP 来自组件的 `end_effector.yaml`。`TacCapGeometry` 与
  `TacCapGripper.load_model()` 使用同一资源；整机安装关系仍由整机 YAML 决定。
- `kinematics.base.rigid_transform()` 统一检查输入刚体矩阵；组件内部 FK 返回值
  直接参与组合，不再在每一层重复检查。
- UMI 与 OpenWAM 的 `xyz + wxyz` 位姿编解码共用
  `policies/xpolicylab/codec.py`，不互相导入对方的 adapter。
- AAC、AutoHorizon、DVAC 继承 `runtime/synchronous.py` 中的
  `SynchronousChunkStrategy`，共用同步请求、提交和动作起始时间处理。
  采样参数与模型诊断信息留在各自的策略中。
- 内置注册表直接声明延迟导入的 `module:factory`。`plugins.load_plugin()`
  统一解析；实际适配参数的工厂继续保留，纯导入转发函数已移除。

## 一体组件与资源归属

目录按组件实际能力组织，不强制把设备拆成裸臂和独立夹爪。YAM 的装配配置将
`end_effector` 设为 `null`：表示没有额外挂接的末端，内置夹爪仍属于 YAM 的第七维。
组件直接提供完整 TCP 运动学；分体设备才组合裸臂法兰模型和工具几何。

YAM 的原模型与求解器已移到 `embodiments/arm/yam/`，臂和自带夹爪的资源一起放在
其 `assets/i2rt/robot_models/` 下；顶层 `assets/` 已移空。旧 YAM 硬件和运动学导入路径已删除；公开接口直接指向组件。Viewer 使用组件提供的显示坐标映射，将归一化夹爪
展开成两个指尖关节；控制仍发送原来的完整 7 维目标。

新的可选实验是 `manimux/configs/experiments/put_bottles/yam_pi05_joint.yaml`，工位模板为
`manimux/configs/local/yam.example.yaml`。构造和模型加载不打开 CAN；连接和运动是显式操作。
接口与迁移边界见 [YAM 一体组件](yam-integrated-component.md)。

Tianji 的资源已经分别放在：

- `embodiments/arm/tianji/assets/`：左右臂 URDF 与 meshes。
- `embodiments/end_effector/taccap/assets/`：末端几何、URDF 与 meshes。
- `embodiments/robot/tianji_taccap/assets/`：整机支架资源。

YAM 采集与旧实验均使用 `type: yam`；控制共享配置迁到
`manimux/configs/embodiment/robot/yam_control.yaml`。旧 `robots/yam/`、左右臂历史 YAML、
原始 CAN 旁路记录及锁探针已移除。RealSense 只保留组件内的一套 SDK 实现，
采集端转换帧格式，独立相机网络服务位于 `camera_server/`。
ManiUniCon 与 mock 机器人实现、注册、示例和独立模拟运行脚本已删除。
采集 GUI 的 `--mock` 模式也已移除；回归测试的设备替身仅放在 `tests/support/`。
旧 Tianji 驱动备份、注册和手动恢复分支已删除；既有传球模板改用 `tianji_taccap`。
Orbbec 的服务端采集实现已迁入 sensor 组件目录，采集 GUI 的专用接口保持原有实现。一级 `policy_adapter/` 已完成实际实现迁移；专用 IK 求解器的统一注入仍需保持各自阈值和 TCP 语义，不在目录迁移中改写。

## 必须保持的行为

Tianji FK/IK 使用官方实现，保留旧版求解约束。控制位姿始终相对各自手臂基座；
公共场景坐标变换只用于 Viewer。末端安装与 TCP 变换仍参与组合运动学。
配置组织调整不得改变动作间隔、执行限位、采集时间戳或真实硬件连接行为。
