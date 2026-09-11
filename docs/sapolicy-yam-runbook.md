# SAPolicy + YAM 接入手册

## 2026-09-10 真机基线

当前瓶子任务使用 `server/teleop50-raw.yaml` 配合
`infra/manimux-braking-h25.yaml`，不是下文早期 ABC 权重示例：

```bash
envs/yam/.venv/bin/manimux serve \
  --config configs/sapolicy/yam/infra/manimux-braking-h25.yaml
```

已有模型服务、相机和 Viewer 时只使用现有服务流程；GUI Prepare 后由操作者点
Start。模型 RAW 预测 50 步，执行前 25 个源时间步内仍有效的部分。关节限速
0.6 rad/s、1.5 rad/s²；夹爪连续控制限速 1/s、12/s²。双臂并行 IK，失败臂减速
保持，另一臂继续。抓取时锁定开始闭合的位姿，到位后闭合，开度稳定且取得新观测后
才继续移动；释放时锁定末端目标，到位后完成张爪，再等待释放后的新观测。
正常 Finish 时双臂同时 Home，时长 5 秒。

操作者最新反馈为“抓的不是很准，但行为没问题”。这是当前可复现的行为基线，
抓取精度仍待单独排查。执行细节见[减速跟踪](braking-execution.md)和
[独立 IK 与释放控制](independent-ik-execution.md)。

清理前的代码以文件形式保存在
`/home/ubuntu/sa/diagnostics/cleanup_20260910/before-cleanup-manimux.tar`，
配套 XPolicyLab 备份为同目录的 `before-cleanup-xpolicylab.tar`。
UMI／Cartesian／协同执行、
direct/debounced 夹爪试验和重复配置已从当前版本移除；完整 direct 执行器及
`manimux-direct-async.yaml` 留作历史对照。恢复某个试验时先将备份解压到独立目录，
避免覆盖当前真机工作目录。

按操作者要求，本轮修改不做 commit 或 push，保留为本地工作区改动。
权重、训练 YAML、
normalizer、DINO 权重和 SpatialAlign Python 源码的 SHA-256 记录在
`configs/sapolicy/yam/server/teleop50-raw.assets.json`，这些大文件不进入 Git。
XPolicyLab 的 SA 预处理修复仍是子仓库中的本地改动；其他模型的待提交内容
保持原样。离线回归命令：

```bash
bash scripts/validation/test_sapolicy_rollout.sh
```

该命令只运行单元测试和 mock runtime，不连接真机、不加载模型权重。

## 当前结论

SAPolicy 经 **XPolicyLab WebSocket** 接入 ManiMux：

- `XPolicyLab/policy/SAPolicy`：薄封装，推理走本机 `~/sa/SpatialAlignPolicy`；
- ManiMux `worker: xpolicylab_ws`：通用 WS 传输；
- ManiMux `adapter: sapolicy_yam`：相机/内参、YAM FK/IK，以及绝对 EE wire → `joint_position ActionChunk`。

## 数据流

```text
YAM joints + named RGB frames
  -> sapolicy_yam: FK + K → additional_info.sapolicy
  -> xpolicylab_ws → XPolicyLab/policy/SAPolicy (SpatialAlign infer)
  -> packed_ee_wire (H,16) absolute EE
  -> sapolicy_yam: measured-state-seeded IK (fail → hold; Timeline 裁过期步)
  -> canonical left_arm/right_arm joint_position ActionChunk
  -> ManiMux Timeline / executor / Safety / Recorder / YAM driver
```

SpatialAlign 代码与依赖保留在独立仓库/环境；ManiMux 不 import torch。

## 启动

```bash
# SpatialAlign venv（加载 3cam_tcp.ckpt）
cd /home/ubuntu/sa/SpatialAlignPolicy && source .venv/bin/activate
pip install -e /home/ubuntu/manimux/XPolicyLab
python /home/ubuntu/manimux/scripts/servers/sapolicy_yam_server.py \
  --config /home/ubuntu/manimux/configs/sapolicy/yam/server/abc-bottles.yaml

# 相机必须 640x480；模型内部再裁到训练分辨率
envs/yam/.venv/bin/manimux-camera-server --config configs/sapolicy/yam/cameras.yaml
envs/yam/.venv/bin/manimux-viewer --robot yam --host 0.0.0.0 --port 8086
envs/yam/.venv/bin/manimux run --config configs/sapolicy/yam/infra/manimux-xpl.yaml
```

权重默认 `3cam_tcp.ckpt`（仓库根目录）。`cfg_file` 必须与该 checkpoint 的训练架构一致。

## 契约

| 项目 | 当前约束 |
|---|---|
| embodiment | 双臂 YAM，每侧 6 arm joints + 1 gripper |
| observation | 命名 RGB + 标定 `3x3 K` |
| DiT 原生动作 | 相对 `[pose18 \| grip2]`：`pos3+rot6d` ×2 + grip ×2 = 20D |
| wire → ManiMux | 绝对 `pos3+quat_xyzw+grip` ×2 = 16D |
| ManiMux action | 两组绝对关节，每组 7 维 |
| depth | 不支持 |
| IK | 失败 hold 上一步；过期步由 Timeline 裁切 |

Wire endpose 是 YAM `grasp_site` / ABC TCP，不再做 RoboTwin 的 0.12 m 前向偏移。

## 本机 mock（无真机）

无权重联调用 `server/mock.yaml`（`dry_run: true`，回放当前 EE）。

```bash
# 缺 mink/i2rt 时脚本会把 IK 退化成 hold-seed
python scripts/validation/sapolicy_yam_mock_run.py

# 已有 envs/yam 时也可拆成两进程
python scripts/servers/sapolicy_yam_server.py --config configs/sapolicy/yam/server/mock.yaml
envs/yam/.venv/bin/manimux run --config configs/sapolicy/yam/infra/mock.yaml
```

## 分层验证

1. 离线：wire / IK 契约与 adapter 单测
2. 本机 mock：`scripts/validation/sapolicy_yam_mock_run.py`
3. GPU：server `--check` 后真实 forward（非 dry_run）
4. 只读 preflight → 短真机
