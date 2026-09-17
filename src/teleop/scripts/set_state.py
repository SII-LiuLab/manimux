#!/usr/bin/env python3
"""Switch one arm's control state -- position / joint impedance / Cartesian
impedance / drag / release / disabled -- without commanding a new target.
position/impedance hold at the current pose (same pattern as
goto_joints.py's post-move hold). Useful for testing a mode switch in
isolation, e.g. feeling impedance compliance by hand.

Read docs/algos.md#impedanceconfig and docs/core.md's `--control-mode
impedance` section before the first --state impedance on a given arm.

--state drag is Marvin_Robot.set_drag_space() layered on an active
impedance mode (drag-teaching), not a cur_state value. Based on
DEMO_PYTHON/showcase_joint_drag_arm_A.py, with two deviations
(docs/scripts.md):
    1. explicit low-stiffness K/D (--drag-stiffness/--drag-damping) before
       set_drag_space -- the demo sends none, which is too stiff to move.
    2. reference rate-limit-tracks the measured pose (--drag-track-rate
       deg/s) instead of copying it verbatim, so K/D has an error to act on.
Exit is set_drag_space dgType=0 before disable().  Alongside the joints,
each drag printout carries the flange pose (xyzabc in the arm's base
frame, mm/deg) -- the controller's realtime stream only feeds back joints,
so it comes from FK on the measured q (Marvin_Kine.fk via ArmIK, no tool
transform: this is the bare flange, not the TCP).

--tool (--state drag/release) registers a mounted tool's mass/center-of-
mass/inertia -- configs/tool/<name>.yaml, drivers.tool_config.ToolConfig
-- via Marvin_Robot.set_tool() before entering torque/CR mode, so the
controller's own gravity-feedforward model (built from the arm's
Mass/MCP/I in robot.ini) accounts for the tool's weight too. This is the
fix for the gravity-sag symptom logged in docs/scripts.md -- the low drag
K/D was never the real cause, an unregistered tool mass was. Omit --tool
to leave tool params untouched (no change from prior behavior). Add a new
tool by dropping a configs/tool/<name>.yaml next to
configs/tool/omnigripper.yaml, not by passing raw numbers on the CLI --
see that file for why (mass/COM need real measurement, not a guess typed
in on the day someone happens to need drag mode).

--state release is the SDK's own zero-force mode (Marvin_Robot.set_state
state=4, STATE_COOP_RELEASE / "协作释放"), not the drag-space machinery
above. DEMO_PYTHON/showcase_collaborative_release.py: gravity-compensated
by the controller itself, no K/D or drag-space call needed -- but it's
all-axis, not directional like --state drag --drag-space X/Y/Z/R. Shares
--tool with --state drag: without the tool registered, the same gravity
sag shows up here too since both modes read the same controller-side
model.

Usage:
    python3 scripts/set_state.py --arm A --state position
    python3 scripts/set_state.py --arm A --state impedance                          # --impedance-type joint by default
    python3 scripts/set_state.py --arm A --state impedance --impedance-type cartesian
    python3 scripts/set_state.py --arm A --state drag                               # joint-space drag (default)
    python3 scripts/set_state.py --arm A --state drag --drag-space X                # Cartesian X-direction drag
    python3 scripts/set_state.py --arm A --state drag --tool omnigripper            # + gravity comp for the registered gripper
    python3 scripts/set_state.py --arm A --state release --tool omnigripper         # SDK-native zero-force free-drive
    python3 scripts/set_state.py --arm A --state disabled
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                     # noqa: E402
from algos.solver_config import load_impedance_config             # noqa: E402
from drivers.arm_driver import (ArmDriver, RobotConnection,       # noqa: E402
                                ERR_CODE_CN, STATE_COOP_RELEASE,
                                STATE_POSITION, STATE_TORQUE,
                                ensure_clear, send_joint_commands)
from drivers.tool_config import load_tool_config                  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# drag_space -> (set_drag_space dgType, set_impedance_type type)
_DRAG_SPACE = {
    'joint': (1, 1), 'X': (2, 2), 'Y': (3, 2), 'Z': (4, 2), 'R': (5, 2),
}

# registered tools -- configs/tool/<name>.yaml, one file per mounted tool
_TOOL_DIR = os.path.join(_REPO_ROOT, 'configs', 'tool')
_TOOL_CHOICES = sorted(f[:-len('.yaml')] for f in os.listdir(_TOOL_DIR)
                       if f.endswith('.yaml')) if os.path.isdir(_TOOL_DIR) else []

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, choices=['A', 'B'])
ap.add_argument('--state', required=True,
                choices=['position', 'impedance', 'drag', 'release', 'disabled'])
ap.add_argument('--impedance-type', choices=['joint', 'cartesian'],
                default='joint',
                help='Impedance mode for --state impedance (default joint).')
ap.add_argument('--impedance-config', default=None,
                help='Impedance parameter YAML path. Defaults to '
                     'configs/solver/impedance_<impedance-type>.yaml.')
ap.add_argument('--drag-space', choices=list(_DRAG_SPACE), default='joint',
                help='Drag space for --state drag: joint (default), '
                     'X/Y/Z for Cartesian translation, R for Cartesian '
                     'rotation.')
ap.add_argument('--drag-stiffness', type=float, default=1.0,
                help='Joint K before --state drag --drag-space joint '
                     '(N*m/deg, all axes, range 0-22). Overrides a stale K '
                     'left by a prior --state impedance call.')
ap.add_argument('--drag-damping', type=float, default=0.3,
                help='Joint D for --state drag --drag-space joint (all '
                     'axes, range 0-1; SDK recommends 0.3).')
ap.add_argument('--drag-track-rate', type=float, default=15.0,
                help='Max deg/s per axis the reference tracks the measured '
                     'pose during --state drag. Uncapped tracking keeps '
                     'the error ~0 so K can\'t resist gravity sag; lower '
                     '= closer to a fixed reference, higher = more freely '
                     'followable.')
ap.add_argument('--tool', choices=_TOOL_CHOICES or None, default=None,
                help='Registered tool config (--state drag/release) -- '
                     'configs/tool/<name>.yaml, fed to Marvin_Robot.set_tool() so '
                     "the controller's gravity feedforward accounts for it. See "
                     'module docstring and configs/tool/omnigripper.yaml. Omit to '
                     'leave tool params untouched (no change from prior behavior).'
                     + ('' if _TOOL_CHOICES else ' No configs/tool/*.yaml found.'))
ap.add_argument('--hz', type=float, default=250.0)
args = ap.parse_args()

DT = 1.0 / args.hz

tool_cfg = None
if args.tool:
    try:
        tool_cfg, tool_cfg_path = load_tool_config(args.tool, args.arm, _REPO_ROOT)
    except ValueError as e:
        raise SystemExit('❌ %s' % e)
    print('工具参数(%s, 臂%s): %s\n%s\n'
         % (args.tool, args.arm, tool_cfg_path, tool_cfg.model_dump()))

impedance_cfg = None
if args.state == 'impedance':
    try:
        impedance_cfg, cfg_path = load_impedance_config(
            args.impedance_type, args.impedance_config, _REPO_ROOT)
    except ValueError as e:
        raise SystemExit('❌ %s' % e)
    print('阻抗(%s)配置: %s\n%s\n'
         % (args.impedance_type, cfg_path, impedance_cfg.model_dump()))

conn = RobotConnection(config.ROBOT_IP)
print('✅ 控制器版本 %s' % conn.version)

drv = ArmDriver(conn, args.arm, config, impedance_config=impedance_cfg)
try:
    print('臂%s 检查状态/清错…' % args.arm)
    ok, st = ensure_clear(conn, drv, args.arm)
    print('臂%s %s cur=%s err=%s q=%s'
         % (args.arm, '✅ 状态干净' if ok else '❌ 清错后仍有故障',
            st['cur'], st['err'], [round(v, 2) for v in st['q']]))

    if args.state == 'disabled':
        # `finally` below always calls drv.disable(); nothing else to do here
        print('臂%s 目标状态：失能' % args.arm)

    elif not ok:
        raise RuntimeError(
            '臂%s 清错未成功（cur=%s err=%s(%s)）—— 不去尝试切模式，'
            '先用 MarvinPlatform GUI 或 tools/wait_ready.py 排查'
            % (args.arm, st['cur'], st['err'],
               ERR_CODE_CN.get(st['err'], '未知')))

    elif args.state in ('position', 'impedance'):
        config.ARM_STATE = (STATE_POSITION if args.state == 'position'
                            else STATE_TORQUE)
        drv.prepare()
        q_hold = list(drv.joints())
        print('✅ 臂%s 已切到%s模式，停在当前构型 %s'
             % (args.arm, args.state, [round(v, 2) for v in q_hold]))
        print('>>> 保持中，按 Ctrl+C 退出（退出后会抱死保持姿态，不会下垂）')
        while True:
            send_joint_commands(conn, {args.arm: q_hold})
            time.sleep(DT)

    elif args.state == 'drag':
        dg_type, imp_type = _DRAG_SPACE[args.drag_space]
        print('⚠️  拖动模式：电机仍带电，人手会直接接触/搬动一个通电的机械臂。'
             '建议两人操作，一人扶臂，一人守在 e-stop 旁。拖动期间本脚本'
             '会以 --drag-track-rate 限速跟随实测位置下发关节指令（见'
             'docs/scripts.md），不是完全被动只读。')

        conn.robot.clear_set()
        conn.robot.set_state(arm=args.arm, state=STATE_TORQUE)
        conn.robot.set_impedance_type(arm=args.arm, type=imp_type)
        if tool_cfg is not None:
            # feeds the controller's own gravity-feedforward model (built from
            # the arm's Mass/MCP/I in robot.ini) so it accounts for the
            # mounted tool too -- fixes gravity sag, see module docstring.
            conn.robot.set_tool(arm=args.arm, kineParams=tool_cfg.kine_params(),
                                dynamicParams=tool_cfg.dynamic_params())
            print('臂%s 工具参数已注册：%s（m=%.3fkg com=%s，供扭矩模式重力前馈使用）'
                 % (args.arm, args.tool, tool_cfg.mass_kg, tool_cfg.com_mm))
        if args.drag_space == 'joint':
            # not in vendor demo; overrides stale K/D from a prior --state impedance
            K = [args.drag_stiffness] * 7
            D = [args.drag_damping] * 7
            conn.robot.set_joint_kd_params(arm=args.arm, K=K, D=D)
            print('臂%s 关节拖动刚度 K=%.2f D=%.2f（每轴同值）'
                 % (args.arm, args.drag_stiffness, args.drag_damping))
        conn.robot.send_cmd()
        time.sleep(0.5)
        st = drv.state()
        if st['cur'] != STATE_TORQUE:
            raise RuntimeError('臂%s 切扭矩模式失败：期望 %s 实得 %s'
                               % (args.arm, STATE_TORQUE, st['cur']))
        # prepare() normally sets this; the raw set_state() call above
        # doesn't, and without it the outer `finally: drv.disable()` silently
        # no-ops (see ArmDriver.disable), leaving the arm energized in torque
        # mode after exit.
        drv.engaged = True

        conn.robot.clear_set()
        conn.robot.set_drag_space(arm=args.arm, dgType=dg_type)
        conn.robot.send_cmd()
        time.sleep(0.2)
        got = conn.subscribe()['inputs'][drv.idx]['drag_sp_type']
        if got != dg_type:
            raise RuntimeError('臂%s 拖动空间设置未生效：下发 %s 回读 %s'
                               % (args.arm, dg_type, got))
        print('✅ 臂%s 已进入拖动模式（%s，dgType=%d 已确认）—— 现在可以用手'
             '拖动。按 Ctrl+C 退出拖动' % (args.arm, args.drag_space, dg_type))
        # FK on the measured q, so the printed pose is where the flange
        # actually is (imported here, not at module level: only this branch
        # needs the SDK kine, and --state disabled shouldn't pay for it).
        from algos.ik_solver import ArmIK                       # noqa: E402
        ik = ArmIK(arm_type=0 if args.arm == 'A' else 1,
                   config_path=config.KINE_CFG)
        # rate-limited so K has a tracking error to resist (see --drag-track-rate)
        max_step = args.drag_track_rate * DT
        q_cmd = list(drv.joints())
        n = 0
        print_every = max(1, int(0.5 / DT))
        try:
            while True:
                q_meas = drv.joints()
                q_cmd = [c + max(-max_step, min(max_step, m - c))
                        for c, m in zip(q_cmd, q_meas)]
                send_joint_commands(conn, {args.arm: q_cmd})
                n += 1
                if n % print_every == 0:
                    x = ik.fk_xyzabc(q_meas)
                    print('  测量 %s  参考 %s\n    法兰 xyz=%s mm  abc=%s °'
                         % ([round(v, 2) for v in q_meas],
                            [round(v, 2) for v in q_cmd],
                            [round(v, 1) for v in x[:3]],
                            [round(v, 1) for v in x[3:]]))
                time.sleep(DT)
        finally:
            print('退出拖动空间…')
            conn.robot.clear_set()
            conn.robot.set_drag_space(arm=args.arm, dgType=0)
            conn.robot.send_cmd()
            time.sleep(0.5)

    elif args.state == 'release':
        print('⚠️  协作释放模式（state=4，SDK 原生重力补偿零力漂浮，'
             'DEMO_PYTHON/showcase_collaborative_release.py）：全轴同时松开阻力'
             '（不像 drag 只在选定方向/关节顺从）。电机仍带电，人手会直接接触/'
             '搬动一个通电的机械臂。建议两人操作，一人扶臂，一人守在 e-stop 旁。'
             '本脚本不下发任何关节指令——重力补偿由控制器自己算。')

        conn.robot.clear_set()
        if tool_cfg is not None:
            conn.robot.set_tool(arm=args.arm, kineParams=tool_cfg.kine_params(),
                                dynamicParams=tool_cfg.dynamic_params())
            print('臂%s 工具参数已注册：%s（m=%.3fkg com=%s，供协作释放模式重力前馈使用）'
                 % (args.arm, args.tool, tool_cfg.mass_kg, tool_cfg.com_mm))
        conn.robot.set_state(arm=args.arm, state=STATE_COOP_RELEASE)
        conn.robot.send_cmd()
        time.sleep(1.0)
        st = drv.state()
        if st['cur'] != STATE_COOP_RELEASE:
            raise RuntimeError('臂%s 切协作释放模式失败：期望 %s 实得 %s'
                               % (args.arm, STATE_COOP_RELEASE, st['cur']))
        # see the matching comment in the drag branch: without this,
        # `finally: drv.disable()` below silently no-ops on exit.
        drv.engaged = True
        print('✅ 臂%s 已进入协作释放模式（重力补偿零力漂浮）—— 现在可以用手'
             '整体拖动。按 Ctrl+C 退出' % args.arm)
        n = 0
        print_every = max(1, int(0.5 / DT))
        while True:
            n += 1
            if n % print_every == 0:
                print('  测量 %s' % [round(v, 2) for v in drv.joints()])
            time.sleep(DT)
except KeyboardInterrupt:
    print('\n[Ctrl+C] 退出')
finally:
    drv.disable()
    conn.close()
    print('已下伺服并断开连接')
