#!/usr/bin/env python3
"""UMI / TacCap follower gripper — thin wrapper over the vendor SDK's ControlLoop.

This shares no code with drivers/gripper.py and should not: that one talks
DM4310 MIT frames over Marvin's end-effector CAN pass-through, this one is a
separate USB serial link per side (CH343 -> MCU -> FDCAN -> motor) and never
touches the arm SDK at all. Practical consequences:

  * No RobotConnection needed. The gripper works with the controller powered
    down, and a controller reboot doesn't disturb it.
  * No 50Hz Python thread. `ControlLoop` is a C++ background thread using
    SubmitPhase.STREAM_LOCKED: it emits exactly one command per received
    motor-status frame, landing in the ~9.86ms the MCU is known to be idle.
    The vendor measured 6000 submits : 6000 frames : 0 missing that way,
    against 156-308 lost frames per run when free-running at 100Hz. A Python
    timer thread would be the second case.
  * Force limits are much smaller. UMI_GRIPPER_MAX_TAU is 1.0 N*m, not the
    OmniGripper's 8.0 -- copying that value across would mean no protection
    at all rather than a conservative one.

Convention: 0 = CLOSED, 1 = OPEN, matching the SDK and matching what UMI
episodes record (drivers/umi_source.py), so recorded jaw values feed
set_target() unconverted. This is the inverse of drivers/gripper.py.

Feedback comes from ControlLoop.observation() only. Do NOT poll
motor.read_status() in a loop -- above ~100Hz it stalls the firmware's own
refresh, which is exactly the pattern drivers/gripper.py uses and another
reason not to reuse it.

See docs/drivers.md#umi_gripperpy.
"""
import os
import sys
import time

_SDK = os.environ.get('TACCAP_SDK',
                      '/home/jw/Desktop/project/TacCap-Gripper/python')
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)

try:
    from xense.taccap import (ControlLoop, FollowerGripper,  # noqa: E402
                              Role, Side, scan_grippers)
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        'xense.taccap 导入失败：%s\n'
        '  期望在 %s 找到（可用 TACCAP_SDK 环境变量覆盖）\n'
        '  该 SDK 不是 pip 包，来自 TacCap-Gripper 仓库的 python/ 目录。'
        % (e, _SDK))

# GripperConfig.flags bit 0 -- set once the gripper's open/close travel has
# been calibrated. Without it position() and set_position() have no meaningful
# scale, and the SDK throws rather than guessing.
CFG_VALID = 0x0001

SIDE_OF_ARM = {'A': Side.Left, 'B': Side.Right}


def _find_follower_in(found, arm, serial=None):
    """Resolve one arm from an already scanned endpoint list."""
    if not found:
        raise RuntimeError(
            '没有发现任何 TacCap 夹爪。依次查：\n'
            '  1) USB 线是否插好（应能看到 /dev/ttyACM* 或 /dev/ttyUSB*）\n'
            '  2) 当前用户是否在 dialout 组\n'
            '  3) 夹爪是否上电')
    if serial:
        for ep in found:
            if ep.firmware_sn == serial:
                return ep
        raise RuntimeError('臂 %s 指定的 SN %r 不在已连接的夹爪里：%s'
                           % (arm, serial,
                              [ep.firmware_sn or '<无>' for ep in found]))

    want = SIDE_OF_ARM[arm]
    cands = [ep for ep in found
             if ep.side == want and ep.role == Role.Follower]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise RuntimeError(
            '没有发现臂 %s（%s 侧）的 follower 夹爪。已连接：%s\n'
            '  若侧别规则判错，在 config.UMI_GRIPPER_SN[%r] 里写死 SN。'
            % (arm, want, [(ep.firmware_sn, str(ep.side), str(ep.role))
                           for ep in found], arm))
    raise RuntimeError(
        '臂 %s（%s 侧）匹配到 %d 个 follower：%s\n'
        '  自动判别不了，在 config.UMI_GRIPPER_SN[%r] 里写死 SN。'
        % (arm, want, len(cands), [ep.firmware_sn for ep in cands], arm))


def find_follower(arm, serial=None):
    """Resolve one arm's gripper to a GripperEndpoints, or raise.

    :param serial: firmware SN to pin to. Strongly preferred over the side
                   rule -- see config.UMI_GRIPPER_SN.
    """
    return _find_follower_in(scan_grippers(), arm, serial)


class UmiGripper:
    """One follower gripper, driven in normalized position (0=closed, 1=open)."""

    def __init__(self, arm, cfg, serial=None):
        self.arm = arm
        self.cfg = cfg
        self.serial = serial if serial is not None \
            else cfg.UMI_GRIPPER_SN.get(arm)
        self.g = None
        self.loop = None
        self.endpoint = None
        self.gripper_config = None
        self.stroke_rad = None      # |max_open_rad - min_open_rad|
        self.fault = None           # set on protection trip; callers must check
        self._target = None

    # ---------- lifecycle ----------

    def connect(self, quiet=False, endpoint=None):
        """Open the link, verify calibration, enable the motor, start the loop.

        ControlLoop.start() seeds its target at the current measured position,
        so enabling never produces a jump -- no manual seed step needed.
        """
        self.endpoint = (endpoint if endpoint is not None
                         else find_follower(self.arm, self.serial))
        self.g = FollowerGripper(mcu_device=self.endpoint.mcu_device)

        gc = self.g.get_gripper_config()
        self.gripper_config = gc
        if not (gc.flags & CFG_VALID):
            raise RuntimeError(
                '臂 %s 的夹爪未标定（GripperConfig.flags=0x%04x）。\n'
                '  归一化位置没有物理意义，拒绝启动。先跑一次上电自标定\n'
                '  （set_auto_cal_config）或厂商标定流程。' % (self.arm, gc.flags))
        self.stroke_rad = abs(gc.max_open_rad - gc.min_open_rad)

        self.g.motor.clear_fault()
        self.g.motor.enable()
        self.loop = ControlLoop(self.g,
                                hz=int(self.cfg.UMI_GRIPPER_HZ),
                                kp=float(self.cfg.UMI_GRIPPER_KP),
                                kd=float(self.cfg.UMI_GRIPPER_KD))
        self.loop.start()
        self._target = self.loop.target

        if not quiet:
            print('夹爪 %s: %s SN=%s  行程 %.3f rad  起始位置 %.3f'
                  % (self.arm, self.endpoint.mcu_device,
                     self.endpoint.firmware_sn or '<无>',
                     self.stroke_rad, self._target))
            print('        限力 kp=%.1f × %.3f×%.3f rad ≈ %.3f N·m'
                  ' (保护阈值 %.2f)'
                  % (self.cfg.UMI_GRIPPER_KP,
                     self.cfg.UMI_GRIPPER_GRIP_MARGIN, self.stroke_rad,
                     self.grip_force_nm(), self.cfg.UMI_GRIPPER_MAX_TAU))
        return self

    def close(self):
        """Stop the loop and de-energize. Safe to call twice."""
        try:
            if self.loop is not None:
                self.loop.stop()
        finally:
            self.loop = None
            if self.g is not None:
                try:
                    self.g.motor.disable()
                except Exception:
                    pass

    # ---------- control ----------

    def grip_force_nm(self):
        """Grip force the margin allows, N*m. None before connect()."""
        if self.stroke_rad is None:
            return None
        return (self.cfg.UMI_GRIPPER_KP * self.cfg.UMI_GRIPPER_GRIP_MARGIN
                * self.stroke_rad)

    def set_target(self, x):
        """Command a normalized jaw position. 0 = closed, 1 = open.

        Clamped to stay within UMI_GRIPPER_GRIP_MARGIN of the measured
        position, which is what bounds grip force -- same principle as
        GRIPPER_MAX_OVERSHOOT_RAD, expressed in normalized units here
        because that is what both the SDK and the recordings speak.
        """
        if self.loop is None:
            raise RuntimeError('夹爪 %s 还没 connect()' % self.arm)
        x = max(0.0, min(1.0, float(x)))
        obs = self.loop.observation()
        if obs.valid:
            m = self.cfg.UMI_GRIPPER_GRIP_MARGIN
            x = max(obs.position - m, min(obs.position + m, x))
        self._target = x
        self.loop.set_target(x)
        return x

    def observation(self):
        return None if self.loop is None else self.loop.observation()

    def position(self):
        """Return normalized position from the ControlLoop observation."""
        obs = self.observation()
        if obs is None or not obs.valid:
            raise RuntimeError('夹爪 %s 观测无效' % self.arm)
        return float(obs.position)

    def check(self):
        """Poll protections. Returns the fault string, or None if healthy.

        Latches: once tripped, stays tripped. Callers treat this exactly like
        GripperThread.fault, so run_teleop's existing check reads the same.
        """
        if self.fault or self.loop is None:
            return self.fault
        obs = self.loop.observation()
        if not obs.valid:
            self.fault = '%s 观测无效（电机未使能或链路未起来）' % self.arm
        elif obs.age_ms > self.cfg.UMI_GRIPPER_STALE_MS:
            self.fault = ('%s 观测过期 %.0fms（上限 %.0f），链路可能断了'
                          % (self.arm, obs.age_ms, self.cfg.UMI_GRIPPER_STALE_MS))
        elif abs(obs.torque) > self.cfg.UMI_GRIPPER_MAX_TAU:
            self.fault = ('%s 力矩 %.3fN·m 超限（上限 %.2f）'
                          % (self.arm, obs.torque, self.cfg.UMI_GRIPPER_MAX_TAU))
        return self.fault


class UmiGrippers:
    """The mounted grippers, keyed by arm. Mirrors GripperThread's surface."""

    def __init__(self, cfg, arms):
        self.cfg = cfg
        self.arms = list(arms)
        self.g = {a: UmiGripper(a, cfg) for a in self.arms}

    @property
    def fault(self):
        for a in self.arms:
            if self.g[a].fault:
                return self.g[a].fault
        return None

    def start(self, quiet=False, ready_timeout_s=3.0):
        """Start all configured grippers after one endpoint scan."""
        if all(self.g[a].loop is not None for a in self.arms):
            return self
        found = scan_grippers()
        endpoints = {a: _find_follower_in(
            found, a, self.g[a].serial) for a in self.arms}
        started = []
        try:
            for a in self.arms:
                self.g[a].connect(quiet=quiet, endpoint=endpoints[a])
                started.append(a)
            deadline = time.monotonic() + float(ready_timeout_s)
            while time.monotonic() < deadline:
                if all((o := self.g[a].observation()) is not None and o.valid
                       for a in self.arms):
                    return self
                time.sleep(0.01)
            missing = [a for a in self.arms
                       if (o := self.g[a].observation()) is None or not o.valid]
            raise RuntimeError('夹爪首帧 motor-status 超时：%s' % missing)
        except Exception:
            for a in started:      # never leave a half-open set energized
                self.g[a].close()
            raise

    def aperture(self):
        """Return normalized positions without issuing serial status reads."""
        return [self.g[a].position() for a in self.arms]

    def set_target(self, arm, x):
        return self.g[arm].set_target(x)

    def observation(self, arm):
        return self.g[arm].observation()

    def check(self):
        for a in self.arms:
            f = self.g[a].check()
            if f:
                return f
        return None

    def stop(self):
        for a in self.arms:
            try:
                self.g[a].close()
            except Exception:
                pass


if __name__ == '__main__':
    import argparse

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    import config                                  # noqa: E402

    ap = argparse.ArgumentParser(
        description='Standalone open/close smoke test for the UMI follower '
                    'gripper -- USB serial only, never touches the arm.')
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--num', type=int, default=1, help='open/close cycles')
    ap.add_argument('--hold', type=float, default=2.0,
                    help='seconds to watch after each move')
    ap.add_argument('--scan', action='store_true',
                    help='Just list connected grippers and exit.')
    args = ap.parse_args()

    if args.scan:
        eps = scan_grippers()
        print('发现 %d 个夹爪：' % len(eps))
        for ep in eps:
            print('  dev=%s  SN=%s  role=%s  side=%s'
                  % (ep.mcu_device, ep.firmware_sn or '<无>', ep.role, ep.side))
        raise SystemExit(0)

    g = UmiGripper(args.arm, config).connect()
    try:
        for i in range(1, args.num + 1):
            print('\n=== 第 %d/%d 次开合 ===' % (i, args.num))
            for label, tgt in (('闭合', 0.0), ('张开', 1.0)):
                print('>>> %s…' % label)
                t0 = time.monotonic()
                while time.monotonic() - t0 < args.hold:
                    g.set_target(tgt)           # re-issued: the margin clamp
                                                # walks the target in gradually
                    if g.check():
                        print('❌ 保护触发：%s' % g.fault)
                        raise SystemExit(1)
                    o = g.observation()
                    print('   pos=%.3f  vel=%+.3f  tau=%+.3f N·m  age=%.0fms'
                          % (o.position, o.velocity, o.torque, o.age_ms))
                    time.sleep(0.2)
        print('\n✅ %d 次开合完成，无保护触发' % args.num)
    finally:
        g.close()
        print('夹爪已失能')
