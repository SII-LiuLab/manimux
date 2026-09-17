# 本地工位绑定

复制 `tianji_taccap.example.yaml` 到 Git 忽略的 `.local/`，填写自己的设备信息。
只有实际安装并需要独立连接的组件才出现在 `robot.components` 中，不用填 null 占位。

```yaml
robot:
  hardware:
    ip: 192.0.2.10  # 仅需要 IP 的共享控制器填写
  components:
    left_end_effector:
      serial: REPLACE_LEFT_GRIPPER
    left_wrist_camera:
      camera_serial: REPLACE_LEFT_CAMERA
services:
  camera:
    endpoint: tcp://127.0.0.1:5556
    request_endpoint: tcp://127.0.0.1:5555
  policy:
    endpoint: ws://127.0.0.1:8560
paths:
  checkpoint: ../checkpoints/pass_ball
```

不同组件可以使用 `channel: can_left` 或 `port: /dev/ttyUSB0`，不强制 IP/序列号。
具体字段须对应组件构造接口；本轮实现的是 Tianji 新整机绑定，YAM 仍走旧入口。
夹爪集成在手臂里、共享同一通信通道时，只绑定手臂即可，不创建空夹爪绑定。
省略 local 绑定不会删除本体组件；独立设备缺少必要参数时，在连接阶段报错。

local 不包含 `execute`、模型动作格式、安装矩阵或 TCP。物理组成在 embodiment 中定义；
动作执行由 experiment 明确选择。未知组件名会报错，避免拼写错误被静默忽略。
路径相对 local 文件解析。CLI `--local` 优先于实验文件中的 `local:`。

完整入口见 [Tianji–TacCap runbook](../../docs/umi-dp-tianji-taccap-runbook.md)。
