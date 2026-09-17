#!/usr/bin/env python3
"""Robot arm driver: Marvin_Robot lifecycle, mode switching, command dispatch.

Scope: how to safely push joint angles into the controller. What angle to
send is retarget/ik_solver's job; whether to send it at all is safety's job.

See docs/drivers.md for design rationale, empirical data, and known pitfalls.
"""
import math
import os
import sys
import time

_SDK = os.environ.get('MARVIN_SDK',
                      '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS  # noqa: E402

# cur_state values — see python_doc_contrl.md section 6
STATE_DISABLED = 0
STATE_POSITION = 1
STATE_PVT = 2
STATE_TORQUE = 3
STATE_COOP_RELEASE = 4
STATE_ERROR = 100

_ARM_IDX = {'A': 0, 'B': 1}


class RobotConnection:
    """Single connection shared by both arms and the gripper."""

    def __init__(self, ip, dry_run=False, quiet=True):
        self.ip = ip
        self.dry_run = dry_run
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        self.connected = False
        self.version = None

        if not self.robot.connect(ip):
            raise SystemExit(
                '连接 %s 失败：端口被占用。\n'
                '  → MarvinPlatform GUI 是否还开着？点 Disconnect 不释放端口，'
                '必须完全退出应用。\n'
                '  → 检查：ss -lunp | grep 4730' % ip)
        self.connected = True
        if quiet:
            self.robot.log_switch('0')
            self.robot.local_log_switch('0')

        # version reads 0 right after a controller restart; not a fault
        _, ver = self.robot.get_param('int', 'VERSION')
        if not ver:
            self.close()
            raise SystemExit('控制器尚未启动完成（版本号读到 0）。\n'
                             '  → 先跑 tools/wait_ready.py')
        self.version = ver

        # confirm the UDP data channel is actually flowing, not just connected
        frames, last = 0, None
        for _ in range(5):
            fs = self.subscribe()['outputs'][0]['frame_serial']
            if fs and fs != last:
                frames, last = frames + 1, fs
            time.sleep(0.01)
        if frames < 3:
            self.close()
            raise SystemExit('实时帧不刷新（5 次采样只变化 %d 次）\n'
                             '  → 控制器仍在启动，或防火墙拦了 UDP' % frames)

    def subscribe(self):
        return self.robot.subscribe(self.dcss)

    def check_and_clear_errors(self):
        self.robot.check_error_and_clear(self.dcss)

    def close(self):
        if self.connected:
            self.robot.release_robot()
            self.connected = False


class ArmDriver:
    """Per-arm mode switching and command dispatch over a shared RobotConnection."""

    def __init__(self, conn, arm, cfg, impedance_config=None):
        if arm not in _ARM_IDX:
            raise ValueError("arm 必须是 'A' 或 'B'")
        self.conn = conn
        self.arm = arm
        self.idx = _ARM_IDX[arm]
        self.cfg = cfg
        self.impedance_config = impedance_config
        self.robot = conn.robot
        self.engaged = False            # True once switched to a motion mode
        self.last_cmd = None
        self.sent = 0
        self.skipped = 0

    # ---------- state ----------

    def state(self):
        s = self.conn.subscribe()
        return {
            'cur': s['states'][self.idx]['cur_state'],
            'cmd': s['states'][self.idx]['cmd_state'],
            'err': s['states'][self.idx]['err_code'],
            'q': list(s['outputs'][self.idx]['fb_joint_pos']),
            'q_cmd': list(s['inputs'][self.idx]['joint_cmd_pos']),
            'frame': s['outputs'][self.idx]['frame_serial'],
            'vel_ratio': s['inputs'][self.idx]['joint_vel_ratio'],
            'acc_ratio': s['inputs'][self.idx]['joint_acc_ratio'],
        }

    def joints(self):
        return self.state()['q']

    # ---------- mode switching ----------

    def prepare(self):
        """Switch to the configured control mode; allows settling time before/after."""
        st = self.state()
        if st['err'] or st['cur'] == STATE_ERROR:
            raise RuntimeError('臂%s 存在错误（状态 %s 错误码 %s），先清错'
                               % (self.arm, st['cur'], st['err']))
        if self.conn.dry_run:
            print('[dry-run] 跳过臂%s 的模式切换（当前状态 %s）'
                  % (self.arm, st['cur']))
            self.engaged = True
            return

        # vel/acc ratio must be set before the mode switch
        self.robot.clear_set()
        self.robot.set_vel_acc(arm=self.arm, velRatio=self.cfg.VEL_RATIO,
                               AccRatio=self.cfg.ACC_RATIO)
        if self.cfg.ARM_STATE == STATE_TORQUE:
            ic = self.impedance_config
            if ic is None:
                raise ValueError(
                    "ARM_STATE=3（扭矩）需要 impedance_config"
                    '（一个 algos.solver_config.ImpedanceConfig）')
            if ic.type == 3:
                raise ValueError(
                    'impedance_config.type=3（力控）ArmDriver 未实现——'
                    '力控走的是不同的调用序列（set_force_control_params/'
                    'set_force_cmd），用 type 1 或 2')
            print('⚠️  臂%s 首次由软件驱动扭矩/阻抗模式（此前只在 GUI 上手动点过）——'
                  '确认周围空旷、手在 Ctrl+C 附近' % self.arm)
            self.robot.set_impedance_type(arm=self.arm, type=ic.type)
            # Both panels are set unconditionally (mirrors the GUI's "Save
            # parameters" saving joint+Cartesian together) -- only
            # whichever `type` was just selected above actually governs
            # runtime behavior; this just means switching type later
            # doesn't need re-sending params.
            self.robot.set_joint_kd_params(arm=self.arm, K=ic.joint_k,
                                           D=ic.joint_d)
            self.robot.set_cart_kd_params(arm=self.arm, K=ic.cart_k,
                                          D=ic.cart_d, type=ic.type)
            if ic.type == 2:
                # End-effector rotation reference -- Cartesian impedance
                # only, no joint-impedance equivalent. set_cart_kd_params
                # above has no parameter for this; the vendor's demo for
                # this scenario (DEMO_C++/showcase_eef_cart_impedance.cpp)
                # always pairs OnSetCartKD_A with OnSetEefRot_A in the same
                # clear_set/send_cmd batch, so this call is required too.
                # See docs/algos.md#impedanceconfig for why fcType=2 (not
                # 0) is used: it's the only value taken from a working demo
                # rather than inferred from a docstring -- the SDK's two
                # docstrings mentioning this parameter (set_imp_cart_
                # state's `rot_type` and set_EefCart_control_params's own
                # `fcType`) don't agree on numbering, and neither
                # documents a "0".
                self.robot.set_EefCart_control_params(
                    arm=self.arm, fcType=ic.rot_type,
                    CartCtrlPara=ic.cart_ctrl_para)
        self.robot.send_cmd()
        time.sleep(0.2)

        # confirm the vel ratio actually took; see docs/drivers.md
        self._check_vel_ratio('set_vel_acc 之后')

        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=self.cfg.ARM_STATE)
        self.robot.send_cmd()
        time.sleep(1.0)                 # give the servo time to respond; don't shorten

        st = self.state()
        if st['cur'] != self.cfg.ARM_STATE:
            raise RuntimeError('臂%s 切模式失败：期望 %s 实得 %s'
                               % (self.arm, self.cfg.ARM_STATE, st['cur']))
        self._check_vel_ratio('set_state 之后')
        self.engaged = True

    def _check_vel_ratio(self, when):
        """Confirm the controller's live vel/acc ratio matches cfg; see docs/drivers.md."""
        st = self.state()
        vr, ar = st['vel_ratio'], st['acc_ratio']
        if not 1 <= vr <= 100:
            print('⚠️  臂%s 速度档回读不可用（%s 读到 vel=%r acc=%r）。'
                  '无法确认控制器实际速度，%s 的限幅前提未经验证。'
                  % (self.arm, when, vr, ar, self.cfg.VEL_RATIO))
            return
        if vr != self.cfg.VEL_RATIO:
            raise RuntimeError(
                '臂%s 速度档不符（%s）：下发 %d%% 回读 %d%%。\n'
                '  上层限幅是按控制器 %d%% 推的，不匹配就不能上机 ——\n'
                '  若是 set_state 冲掉了，把 set_vel_acc 挪到切模式之后重发。'
                % (self.arm, when, self.cfg.VEL_RATIO, vr, self.cfg.VEL_RATIO))
        print('臂%s 速度档已确认（%s）：vel %d%% / acc %d%%'
              % (self.arm, when, vr, ar))

    def disable(self):
        """De-energize this arm; blocks until confirmed. Every exit path must call this."""
        if self.conn.dry_run or not self.engaged:
            self.engaged = False
            return
        try:
            for _ in range(3):
                self.robot.clear_set()
                self.robot.set_state(arm=self.arm, state=STATE_DISABLED)
                self.robot.send_cmd()
                time.sleep(0.3)
                try:
                    if self.state()['cur'] == STATE_DISABLED:
                        return
                except Exception:
                    return              # connection already gone; can't confirm, and must not raise here
            print('⚠️  臂%s 下伺服未确认，电机可能仍带电 —— 请跑 '
                  'tools/wait_ready.py 检查状态，必要时直接断电' % self.arm)
        finally:
            self.engaged = False

    def stop(self):
        """Soft e-stop, faster than disable(); for exception paths."""
        if not self.conn.dry_run:
            try:
                self.robot.stop_running(self.arm)
            except Exception:
                pass


# From TJ_FX_ROBOT_CONTRL_SDK/README.md's err_code table -- only the ones
# worth calling out by name; anything else just prints the raw code.
ERR_CODE_CN = {
    1: '总线拓扑异常', 2: '伺服故障', 3: 'PVT异常',
    4: '请求进位置失败', 5: '进位置失败', 6: '请求进扭矩失败',
    7: '进扭矩失败', 8: '请求上伺服失败', 9: '上伺服失败',
    10: '请求下伺服失败', 11: '下伺服失败', 12: '内部错',
    13: '急停（Emcy）', 14: '浮动基座配置但无IMU硬件', 15: 'PDO工作不正常',
}


def ensure_clear(conn, driver, arm, retries=5, wait=0.3):
    """Check state; if faulted (cur==STATE_ERROR or err_code!=0), clear and
    re-read until confirmed clean or retries exhausted. conn.check_and_clear_errors()
    only fires clear_error() once with no confirmation -- not enough right
    after an e-stop, where err_code=13 (Emcy) can still be latched even
    after the physical button is released. See README.md's err_code table
    and its own note that an e-stop auto-disables the servo, so it must be
    cleared before the servo can be re-enabled."""
    for attempt in range(retries):
        st = driver.state()
        if not st['err'] and st['cur'] != STATE_ERROR:
            return True, st
        desc = ERR_CODE_CN.get(st['err'], '未知错误码')
        print('  臂%s 状态 cur=%s err=%s(%s) —— 清错 (第%d次)'
             % (arm, st['cur'], st['err'], desc, attempt + 1))
        conn.robot.clear_error(arm)
        time.sleep(wait)
    st = driver.state()
    return (not st['err'] and st['cur'] != STATE_ERROR), st


def send_joint_commands(conn, cmds):
    """Dispatch multiple arms' joint commands inside one clear_set/send_cmd pair.

    :param cmds: {'A': [7 angles], 'B': [...]}, degrees
    :return: whether a command was actually sent (always False under dry_run)
    """
    if not cmds:
        return False
    if conn.dry_run:
        return False
    conn.robot.clear_set()
    for arm, q in cmds.items():
        conn.robot.set_joint_cmd_pose(arm=arm, joints=list(q))
    conn.robot.send_cmd()
    return True


def move_to_joints(conn, drivers, targets, speed_deg_s=8.0, hz=250.0,
                   max_err_deg=5.0, settle_deg=1e-3, quiet=False):
    """Move several arms to target joint configurations together, all axes
    sharing one cosine ease-in/ease-out time base (S-curve), not a constant
    rate. Every axis's fraction-of-travel is the same function of elapsed
    time, so the move is a straight line in joint space (no axis parks at
    its target early while others are still catching up -- see
    docs/drivers.md) and velocity is 0 at both endpoints instead of
    stepping instantly to full speed (no jerk spike at start/stop).

    speed_deg_s bounds the PEAK per-axis rate, reached at the midpoint of
    the move, not the average -- so total time is pi/2 (~1.57x) the old
    constant-rate estimate for the same distance. Deliberate: this is the
    homing path, where slower-but-smoother is the right tradeoff, not
    teleop follow.

    :param drivers: {'A': ArmDriver, ...}, must already be prepare()'d into position mode
    :param targets: {'A': [7 angles], ...}, only arms present here are moved
    :return: (ok, reason, final commanded configuration dict)

    The 3rd return value is where the arm actually stopped, which may differ
    from targets on interrupt/timeout — see docs/drivers.md before using it
    to "hold position" after the call.
    """
    if conn.dry_run:
        return True, 'dry_run', dict(targets)
    if not targets:
        return True, 'nothing_to_do', {}

    dt = 1.0 / hz
    q0 = {a: list(drivers[a].joints()) for a in targets}    # start from measured pose
    delta0 = {a: [t - c for t, c in zip(targets[a], q0[a])] for a in targets}
    dmax = max(max(abs(d) for d in delta0[a]) for a in targets)
    if dmax <= settle_deg:
        return True, 'already_there', q0

    # peak rate (at the midpoint of the cosine ramp) equals speed_deg_s
    duration = (math.pi / 2.0) * dmax / speed_deg_s
    if not quiet:
        print('归位：最大单轴 %.1f°，峰值 %.0f°/s（S形加减速，各轴同步）约需 %.0f 秒（Ctrl+C 可中断）'
              % (dmax, speed_deg_s, duration))

    # timeout fallback: 2x nominal duration + 5s, bounds an axis that never converges
    deadline = time.monotonic() + duration * 2.0 + 5.0
    q_cmd = {a: list(q0[a]) for a in targets}
    n = 0
    try:
        while True:
            t = n * dt
            frac = 1.0 if t >= duration else 0.5 * (1.0 - math.cos(math.pi * t / duration))
            for a in targets:
                q_cmd[a] = [q0[a][i] + delta0[a][i] * frac for i in range(7)]

            send_joint_commands(conn, q_cmd)
            n += 1

            if frac >= 1.0:
                break
            if time.monotonic() > deadline:
                return False, 'timeout', q_cmd

            if n % 25 == 0:                     # check tracking every 0.1s
                for a in targets:
                    err = max(abs(x - y) for x, y
                              in zip(q_cmd[a], drivers[a].joints()))
                    if err > max_err_deg:
                        return (False, '臂%s 跟随偏差 %.2f° 超限（撞到东西？）'
                                % (a, err), q_cmd)
                if not quiet and n % 500 == 0:  # progress print every 2s
                    print('  已完成 %.0f%%' % (frac * 100))
            time.sleep(dt)
    except KeyboardInterrupt:
        return False, 'interrupted', q_cmd
    return True, 'ok', q_cmd


class ControlLoop:
    """Fixed-rate control loop; business logic is injected via the step callback.

    step(dt) returns {'A': q, ...}; None or empty means don't send this tick.
    Owns timing, jitter stats, and unconditional disable on exit.
    """

    def __init__(self, conn, drivers, hz):
        self.conn = conn
        self.drivers = drivers          # {'A': ArmDriver, ...}
        self.period = 1.0 / hz
        self.running = False
        self.stats = {'iters': 0, 'sent': 0, 'overruns': 0,
                      'max_jitter_ms': 0.0}

    def run(self, step, duration=None, on_exit=None, before_disable=None):
        """:param before_disable: called after the loop stops, before disabling
                                  servos (homing hooks in here). Runs even if
                                  it raises.
        """
        self.running = True
        t0 = time.perf_counter()
        next_t = t0
        prev = t0
        try:
            while self.running:
                now = time.perf_counter()
                if duration and (now - t0) >= duration:
                    break
                dt = now - prev
                prev = now

                cmds = step(dt)
                if cmds and send_joint_commands(self.conn, cmds):
                    self.stats['sent'] += 1
                self.stats['iters'] += 1

                next_t += self.period
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    # missed the tick: reset baseline instead of bursting to catch up
                    self.stats['overruns'] += 1
                    self.stats['max_jitter_ms'] = max(
                        self.stats['max_jitter_ms'], -slack * 1e3)
                    next_t = time.perf_counter()
        except KeyboardInterrupt:
            print('\n[Ctrl+C] 停止')
        finally:
            self.running = False
            if before_disable:
                # must not skip disable() below, even on a 2nd Ctrl+C during homing
                try:
                    before_disable()
                except BaseException as e:      # noqa: BLE001
                    print('⚠️  归位阶段异常（%s: %s），继续下伺服'
                          % (type(e).__name__, e))
            for d in self.drivers.values():
                d.disable()
            if on_exit:
                on_exit()
        return self.stats


if __name__ == '__main__':
    print(__doc__)
    print('本模块需要真机，不提供离线自测。')
    print('用法见 run_teleop.py；无机器人时用 replay.py 走 dry-run 全链路。')
