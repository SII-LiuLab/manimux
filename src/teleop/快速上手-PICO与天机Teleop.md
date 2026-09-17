# 快速上手：PICO & 天机 Teleop Pipeline

> 三段独立验证：**机器人 → PICO → Teleop**。

| |   路径 |
|---|---|
| Teleop 代码 | `~/Desktop/project/teleop` |
| 机器人 SDK | `~/Downloads/TJ_FX_ROBOT_CONTRL_SDK` |
| 计划与实测记录 | `~/Desktop/project/tianji_teleop_plan.md` |


**运行前注意**

1. **顺序不能跳** —— `roboticsservice` 必须先于头显应用启动。
2. **一个进程只能有一个机器人连接** —— SDK 是 UDP 独占端口，夹爪也走这条连接。
3. **MarvinPlatform GUI 必须完全退出** —— 点 Disconnect **不释放端口**。


---

# 1. 机器人

### 1.1 物理准备

机器人上电 → **手臂举空** → **周围清场 （重要）** → **人手放 E-stop**。

### 1.2 确认没有程序占着连接

```bash
ps aux | grep -E "MarvinPlatform|run_teleop|probe_gripper" | grep -v grep
```

**通过**：无输出。有 MarvinPlatform 就完全退出它（注意不是点 Disconnect）。

### 1.3 确认有线网卡有 IP

```bash
ip -4 addr show enp7s0 | grep inet
```

**通过**：`inet 192.168.1.100/24`

无输出 → `nmcli con up "Profile 1"`。机器人断电后这条链路会掉 （正常来说机器人上电会自动连接）

### 1.4 连通性

```bash
ping -c 3 192.168.1.190
```

**通过**：0% packet loss，延迟 < 1 ms（有线应在 0.1 ms 量级）。

### 1.5 控制器就绪

```bash
cd ~/Downloads/TJ_FX_ROBOT_CONTRL_SDK && python3 tools/wait_ready.py
```

**通过**：
```
✅ 控制器就绪   版本 100343009   实时帧 5/5
   臂A: 状态=0 错误码=0  OK
   臂B: 状态=0 错误码=0  OK
```

刚上电时版本会读到 0、实时帧不刷新，此时末端 CAN 不通 —— 这是**正常的启动过程**，
不是硬件故障，等它就行。

### 1.6 夹爪探活（只读，可跳过）

```bash
python3 tools/probe_gripper_alive.py
```

**通过**：两臂 MST_ID ✅、控制模式 = 1 (MIT) ✅。只读寄存器，**夹爪不会动**。

---

# 2. PICO

### 2.1 （可选）用 PC 自建 5 GHz 热点

**什么时候需要**：2.6 的 `net_check.py` 显示最大 age > 0.25 s，或实跑中出现秒级掉线。

手机热点把手机夹在控制链路正中间（主动省电、WiFi 芯片一芯二用、后台抢资源），
是已知瓶颈。自建热点把手机移出链路：`PICO ──WiFi──▶ PC ──有线──▶ 机器人`。

> ⚠️ **动手前先读完**：热点一起来 PC 就**没有外网**了（`enp7s0` 是机器人专用网， 不提供外网），tailscale 也会断。要查的资料先查完。

```bash
# 1) 配置 —— 此步不断网
nmcli con modify Hotspot \
  802-11-wireless.band a \
  802-11-wireless.ssid teleop5g \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk 'teleop12345'

# 2) 启用 —— ⚠️ 此步断开当前 WiFi、失去外网
nmcli con up Hotspot

# 3) 确认
nmcli -t -f NAME,DEVICE,STATE con show --active | grep Hotspot   # 应为 Hotspot:wlp8s0:activated
ip -4 addr show wlp8s0 | grep inet                               # 应为 10.42.0.1
```

`band a` = 5 GHz，`bg` = 2.4 GHz。密码至少 8 位，可改成你想要的。

**建好之后**：2.2 要记的 IP 就是 **10.42.0.1**，PICO 连 `teleop5g`、应用里也填 10.42.0.1。

**若 5 GHz 起不来**（Realtek 驱动在某些监管域下 AP 模式用不了 5 GHz）：
```bash
nmcli con modify Hotspot 802-11-wireless.band bg && nmcli con up Hotspot
```
退到 2.4 GHz **仍然是净胜** —— 手机已被移出链路，少一跳转发、少一个省电设备。

> 本机**没装 `iw`**，`iwconfig` 对 `rtw89` 驱动也不适用（报 `no wireless extensions`）。
> 所以别用 `iw dev wlp8s0 info` 自查频段 —— 以 2.6 `net_check.py` 的实测数字为准。

**用完恢复外网**：见 4.4。

### 2.2 确认 PC 面向 PICO 的网卡 IP

```bash
ip -4 addr show wlp8s0 | grep inet
```

**记下这个 IP**，2.3 和 2.4 两处都要用，必须一致。

> 别用 `enp7s0` 的 `192.168.1.100` —— 那是机器人的有线网，PICO 不在那个网段。

### 2.3 启动服务（必须先于头显应用）

```bash
pkill -f RoboticsServiceProcess
cd ~/Desktop/project/teleop && ./run_pico_service.sh <2.2 的 IP>
```

后台跑：
```bash
nohup ./run_pico_service.sh <IP> > /tmp/picosvc.log 2>&1 &
```

**通过**：
```bash
ss -tlnp | grep 63901        # 应看到 LISTEN 在 <IP>:63901
```

> 必须用 `run_pico_service.sh` 而不是默认的 `runService.sh`：这台机器有多块网卡，
> 服务会自己挑错网卡，表现是"App 填对了 IP 依然 TCP connection failed"。
> 这个脚本用 `LD_PRELOAD` 强制绑定指定 IP。

### 2.4 PICO 端

1. 头显与 PC 连**同一个 WiFi**（自建热点则连 `teleop5g`）
2. 应用里填 PC IP = **2.2 的 IP**
3. 打开 **Tracking - Controller** 和 **Send**
4. **戴上头显** —— 放桌上追踪会停（见下）

### 2.5 验证数据    (首次启动别跳这步)

```bash
cd ~/Desktop/project/teleop && python3 xr_source.py --dump --duration 10 --hz 2
```

**三条判据，缺一不可**：

| 判据 | 通过 |
|---|---|
| 帧率 | ~72 Hz，`parse_errors` = 0 |
| **位姿数值在变** | 动手柄时 L/R 的数字跟着变 |
| 按键有反应 | 捏 grip / trigger 时数值变化 |

<br>

> ⚠️ **已经踩过的坑**：头显放下或没电时，SDK 仍以 72 Hz 推送**最后已知位姿** ——
> 帧率、丢包、解析错误**全部正常**，只有内容是死的。
> 判据：连续上千帧位姿完全相同（到小数点后 3 位）= 追踪没在跑。
> 拿这种数据跑 dry-run 会得到"IK 100%、零钳位"的**假通过**。

### 2.6 量链路质量

```bash
python3 net_check.py 20
```

| 最大 age | 判断 |
|---|---|
| < 0.15 s | 好 |
| < 0.25 s | 可用（`XR_STALE_S` = 0.25，超了机器人冻结） |
| > 0.25 s | 差，回 2.1 自建热点 |


---

# 3. Teleop

### 3.1 dry-run

```bash
cd ~/Desktop/project/teleop && python3 run_teleop.py --dry-run --arms A --duration 60
```

**能验证**：XR → 重定向 → IK → 安全闸 的 250 Hz 链路、循环抖动、`subscribe()` 实时性。

**⚠️ 验证不了**（dry_run 下这三段代码直接 return，压根没执行）：
模式切换、真实下发、下伺服。**它们的首次执行就是第一次去掉 `--dry-run` 那一跑。**

**⚠️ 别看它的安全闸通过率**：不下发 → 机器人不动 → `q_meas` 冻结 →
指令一偏离起始构型超过 5° 就必然每帧触发 `tracking_error`。这个数字**没有意义**。

### 3.2 真实下发

```bash
python3 run_teleop.py --arms A --scale 0.2 --duration 60
```

**看到 `臂A 就绪，当前构型 [...]`** = 模式切换成功（`set_vel_acc` → `set_state(1)` →
1 s 等待 → 状态回读全过）。

**操作**：
- **按住 grip** = 跟随（按下瞬间锁存手与机器人位姿，之后按相对位移映射）
- **松开 grip** = 立即脱开，机器人**保持不动**
- 松开再按住会**重新锁存** —— 手伸远了就松开、收回、再按住，像鼠标提起再放下

**保护触发后必须松开 grip 才能重新进入**（按着不放会被闭锁拒绝）。

### 3.3 常用参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--scale` | 0.2 起步 | 手动 30 cm → 末端 6 cm。**别超 0.35**，腕部 J6 限位仅 ±60° |
| `--arms` | `A` / `AB` | A = 左手柄，B = 右手柄。**双臂同帧下发尚未验证** |
| `--duration` | 秒 | 不给就一直跑，Ctrl+C 退出 |
| `--gripper` | — | 启用夹爪。**`gripper.py` 尚无真机验证**，别和手臂一起首跑 |

按键绑定：`grip` = 离合（手臂），`trigger` = 夹爪，左摇杆 = 臂角。

### 3.4 中止条件

**任一保护连续触发，或末端行为与手不符 → 立即松开 grip。**

以下**不是故障**：
- 顶到限位裕度还继续推 → 机器人停住（裕度在正常工作）
- 网络断流 → `xr_stale` → 机器人停住（失效模式是"停"不是"跑飞"）

---

# 4. 收工

### 4.1 确认无程序占用机器人

```bash
ps aux | grep -E "run_teleop|probe_gripper|wait_ready" | grep -v grep
```

**通过**：无输出。正常退出的程序已由 `finally` 下伺服。

### 4.2 机器人断电

确认 4.1 无输出后再断电。

### 4.3 停 PICO 服务

```bash
pkill -f RoboticsServiceProcess
```

### 4.4 恢复网络

```bash
nmcli con down Hotspot          # 如果 2.1 建了热点
nmcli con up SII.1X             # 恢复公司网与外网
```

> 热点配置会**保留**在系统里，下次直接 `nmcli con up Hotspot` 即可，不用重新配。


**兜底**：`SII.1X` 和手机热点的 `autoconnect` 都是 `yes`，通常会自动重连。
真卡住了：`nmcli radio wifi off && nmcli radio wifi on`。


### 4.5 PICO 端

关闭应用（否则头显会持续发数据耗电）。

---

# 5. 故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 连接机器人失败：端口被占用 | MarvinPlatform GUI 还开着 | **完全退出**应用，点 Disconnect 不释放端口 |
| 控制器版本读到 0、实时帧不刷新 | 刚上电，仍在启动 | 等 `wait_ready.py` 通过，不是故障 |
| ping 不通 192.168.1.190 | `enp7s0` 没 IP | `nmcli con up "Profile 1"` |
| App 填对 IP 仍 TCP connection failed | 服务绑到了错的网卡 | 用 `run_pico_service.sh <IP>`，别用默认 `runService.sh` |
| 帧率正常但**位姿一动不动** | 头显没戴/没电，推的是最后已知位姿 | 戴上头显、唤醒手柄。**上机前必查** |
| 有 TCP 连接但位姿一帧不来 | PICO 断电重连后留下**僵尸 ESTAB 连接** | `pkill -f RoboticsServiceProcess` 后重启服务 |
| 手感发顿、频繁 `xr_stale` | 网络断流 | 跑 `net_check.py`；根治靠独立 AP / 有线 |
| `tracking_error` 擦边触发（5.0x°） | `VEL_RATIO=10%` 跟不上 `CART_MAX_SPEED_MM_S=250` | 提速度比或降笛卡尔限速。**别只放宽阈值** |
| 拒解风暴 / IK 成功率低 | scale 过大，腕部 J6 顶限位 | 降到 0.2~0.35 |
| 机器人不动，一直 `clutch_released` | 没按住 grip，或保护闭锁未解除 | 按住 grip；触发过保护要先松开再按 |

---

# 附：一次完整流程的命令速查

```bash
# ---- 1. 机器人 ----
ps aux | grep -E "MarvinPlatform|run_teleop" | grep -v grep    # 应无输出
ip -4 addr show enp7s0 | grep inet                             # 应有 192.168.1.100
ping -c 3 192.168.1.190
cd ~/Downloads/TJ_FX_ROBOT_CONTRL_SDK && python3 tools/wait_ready.py

# ---- 2. PICO ----
# （可选）自建 5GHz 热点，断外网：nmcli con up Hotspot   → IP 变 10.42.0.1
ip -4 addr show wlp8s0 | grep inet                             # 记下 IP
pkill -f RoboticsServiceProcess
cd ~/Desktop/project/teleop
nohup ./run_pico_service.sh <IP> > /tmp/picosvc.log 2>&1 &
ss -tlnp | grep 63901
# → 头显：连同一 WiFi、填 <IP>、开 Tracking-Controller + Send、戴上头显
python3 xr_source.py --dump --duration 10 --hz 2               # 位姿必须在变
python3 net_check.py 20                                        # 最大 age < 0.25s

# ---- 3. Teleop ----
python3 run_teleop.py --dry-run --arms A --duration 60
python3 run_teleop.py --arms A --scale 0.2 --duration 60

# ---- 4. 收工 ----
ps aux | grep run_teleop | grep -v grep                        # 应无输出 → 断电
pkill -f RoboticsServiceProcess
nmcli con down Hotspot; nmcli con up SII.1X
```

---

*最后更新：2026-08-13（Session D 首次上机后）。实测数据与完整踩坑记录见
`~/Desktop/project/tianji_teleop_plan.md` 第 4.7 节。*
