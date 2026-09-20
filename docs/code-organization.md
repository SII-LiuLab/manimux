# ManiMux 代码组织

本文件区分目标布局与当前迁移状态。当前先重组 Tianji–TacCap；YAM 和其他历史集成
不因目录调整而切换实现。XPolicyLab 是独立 Git 仓库。

## 目标布局

```text
manimux/
├── XPolicyLab/policy/<POLICY>/  # 学习模型实现与模型服务
├── configs/
│   ├── embodiment/             # 可复用组件、整机装配
│   ├── policy/                 # 模型服务和 policy adapter 参数
│   ├── experiments/            # 实验选择和执行参数
│   └── local/                  # 工位模板；个人绑定放到忽略的 .local/
├── scripts/servers/            # 服务启动入口；调用各自模块的实现
└── src/manimux/
    ├── embodiments/
    │   ├── arm/                # 机械臂 SDK、官方运动学、组件资源
    │   ├── end_effector/       # 末端执行器 SDK、工具几何、组件资源
    │   ├── sensor/             # 传感器 SDK、采集实现、组件资源
    │   └── robot/              # 整机组件管理、RobotModel、整机资源
    ├── kinematics/             # 公共运动学接口、组合算法、几何工具
    ├── policies/               # policy adapter、worker、XPolicyLab 客户端
    ├── server/                 # 独立服务的启动与网络接口
    │   └── sensor/             # 按设备类型组织的传感器服务
    │       └── taccap/         # 相机服务、请求/订阅客户端及 runtime 数据源
    ├── runtime/                # 推理调度、时间线、执行器和必要保护
    ├── viewer/                 # 实时展示与场景摆放
    ├── evaluation/
    ├── collection/
    ├── recording/
    ├── types.py                # 跨模块交换的状态、命令、图像、动作数据
    ├── clock.py
    ├── cli.py                  # 命令行主入口、YAML 读取与实验配置组织
    ├── session.py              # 运行会话协调
    └── __main__.py
```

这个布局不要求单独的 Python 配置模块，也不要求为每份 YAML 定义配置类。
`read_yaml()` 只把 YAML 读成字典，与业务字段无关。实验引用、相对路径和 local
合并遵循明确规则，由入口组织；各组件解释自己的参数并完成构造。
不自动加载任意路径字符串，读取配置也不连接硬件。

注册功能放在所属模块：根据配置中的名称选择对应实现。目标不依赖第三方包的
entry point 自动发现；现有动态导入配置须先迁移，再移除通用插件加载器。

服务启动入口可以集中，具体实现按职责放置。学习模型和模型服务仍在 XPolicyLab，
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
  sensor 生命周期本轮保持原行为；由独立相机服务持有物理相机的目标分工尚需收敛。
- 这里移除的是原顶层配置类及其调用；YAM 采集 GUI 的工位数据类、Viewer 面板
  配置及官方运动学参数不属于这个顶层实验配置接口，未做无关重写。
- 顶层 `robots/`、`sensors/`、`end_effectors/` 已移除。相机服务和客户端迁入
  `server/sensor/taccap/`，保留 ZMQ REP/PUB 协议；`embodiments/sensor/` 只保留
  公共基类、离线 mock 和 TacCap 物理采集实现。
- Tianji 整机只从 `embodiments/robot/tianji_taccap` 装配和运行。
- Tianji Viewer 已通过 `RobotView` 使用 `RobotModel`，显示配置位于
  `viewer/robots/tianji/viewer.yaml`；消息直接保留组名。YAM 显示适配待后续迁移。
  入口与配置见 [Viewer](viewer.md)。

## 共用实现的位置

- YAML 整机装配只有 `RobotModel.from_config()`；原先仅供测试使用的
  `build_tianji_taccap_kinematics()` 与硬编码安装矩阵已删除。测试同样读取装配 YAML。
- TacCap 默认 TCP 来自组件的 `end_effector.yaml`。`TacCapGeometry` 与
  `TacCapGripper.load_model()` 使用同一资源；整机安装关系仍由整机 YAML 决定。
- `kinematics.base.rigid_transform()` 统一检查输入刚体矩阵；组件内部 FK 返回值
  直接参与组合，不再在每一层重复检查。
- UMI 与 OpenWAM 的 `xyz + wxyz` 位姿编解码共用
  `integrations/xpolicylab/obs_codec.py`，不互相导入对方的 adapter。
- AAC、AutoHorizon、DVAC 继承 `runtime/synchronous.py` 中的
  `SynchronousChunkStrategy`，共用同步请求、提交和动作起始时间处理。
  采样参数与模型诊断信息留在各自的策略中。
- 内置注册表直接声明延迟导入的 `module:factory`。`plugins.load_plugin()`
  统一解析；实际适配参数的工厂继续保留，纯导入转发函数已移除。

## 为什么当前还有顶层 assets

`src/manimux/assets/` 目前只剩 `i2rt/robot_models/` 中的 YAM 与 linear_4310
模型。YAM 的历史运动学与 Viewer 路径不在本轮 Tianji 整机重构范围，所以在
“YAM 暂不迁移”的范围下保留。它不是目标架构中的公共本体资源目录。

Tianji 的资源已经分别放在：

- `embodiments/arm/tianji/assets/`：左右臂 URDF 与 meshes。
- `embodiments/end_effector/taccap/assets/`：末端几何、URDF 与 meshes。
- `embodiments/robot/tianji_taccap/assets/`：整机支架资源。

将来迁移 YAM 时，再同步调整资源归属、调用路径、打包和许可证说明，最后移除
顶层 assets。不要只移动文件而留下失效的运动学或 Viewer 默认路径。

## 必须保持的行为

Tianji FK/IK 使用官方实现，保留旧版求解约束。控制位姿始终相对各自手臂基座；
公共场景坐标变换只用于 Viewer。末端安装与 TCP 变换仍参与组合运动学。
配置组织调整不得改变动作间隔、执行限位、采集时间戳或真实硬件连接行为。
