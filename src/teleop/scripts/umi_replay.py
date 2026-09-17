#!/usr/bin/env python3
"""Replay a recorded UMI episode on Tianji — both arms + both follower grippers.

Structurally this is run_teleop.py with the live XR source swapped for a
recorded one: same ArmChannel pipeline (retarget -> IK -> safety gate ->
driver), same ControlLoop, same homing-before-disable discipline. Only
three things differ:

  * frames come from drivers/umi_source.py instead of drivers/xr_source.py,
    pre-interpolated to CONTROL_HZ;
  * the retargeter takes its relative motion in the gripper's own EE frame
    (delta_frame='ee') instead of live teleop's world-referenced delta, so
    no axis map is applied at all. This is the representation the policy
    pipeline trains on (T_cur^-1 @ T_tgt, output_pose_frame=current_ee) and
    the one deployment executes -- replaying any other way would rehearse a
    trajectory nobody is going to run. It also means kine_offset's ROTATION
    now affects the result; see algos/retarget.py's module docstring;
  * --solver defaults to 'diff' (algos/diff_ik.py) rather than run_teleop.py's
    'analytic'. Nobody is in the loop to correct a rejected frame here, and
    diff-IK's rate-limited steps turn rejection into bounded tracking lag.
    See docs/scripts.md#umi_replaypy. --solver analytic restores the old
    behaviour.

Relative, not absolute. The recorded poses are anchored to wherever the
headset was when the VR app started. Frame 0 latches onto each arm's
*measured* pose and everything after is followed relatively, so the arms
start from where they already are and never jump to a recorded absolute
position. There is nothing to recalibrate between recording sessions.

Gripper values pass through unconverted: the recording and the TacCap
follower both use 0 = closed, 1 = open (drivers/umi_gripper.py).

See docs/scripts.md#umi_replaypy.

Usage:
    # 1. no robot at all -- always start here
    python3 replay.py --umi ~/Downloads/replay_test --hand left

    # 2. robot connected, nothing sent (validates prepare/state/tracking)
    python3 scripts/umi_replay.py ~/Downloads/replay_test --dry-run

    # 3. arms move, grippers untouched
    python3 scripts/umi_replay.py ~/Downloads/replay_test --speed 0.3

    # 4. arms + grippers
    python3 scripts/umi_replay.py ~/Downloads/replay_test --gripper

    # single arm: B is parked at all-zero joints first, then A goes to
    # UMI_START_JOINTS and only A's recorded motion is replayed
    python3 scripts/umi_replay.py DIR --arms A --goto-start --speed 0.3

    --arms A          one arm only; every other arm is zeroed first
                      (--no-idle-zero keeps it where it is)
    --scale 0.5       shrink the motion (1.0 = reproduce the demo)
    --vel-ratio 65    clears the rate clamp this demo hits at 55
    --frames 0:144    replay one span (recorded frame indices, as printed by
                      scripts/umi_filter.py)
    --solver analytic use ik_solver.ArmIK instead of the diff-IK default
    --viewer          mirror to universal_viewer
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from core.arm_channel import ArmChannel                          # noqa: E402
from core.homing import home_arms                                # noqa: E402
from drivers.arm_driver import (ArmDriver, ControlLoop,              # noqa: E402
                                RobotConnection, ensure_clear,
                                move_to_joints)
from algos.solver_config import load_diff_ik_config              # noqa: E402
from algos.tool_frame import ToolFrame                          # noqa: E402
from drivers.umi_source import (HANDS, load_episode,           # noqa: E402
                                resample, slice_frames)
from viewer_bridge import ViewerBridge                           # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_OF_ARM = {v: k for k, v in config.ARM_OF_HAND.items()}

# Where an arm that is NOT being replayed is parked before the run starts.
# All-zero joints, not config.HOME_JOINTS: this is the "归零" pose, and it is
# a different physical configuration from the parking pose -- do not swap one
# for the other. Only used when --arms selects a subset, so the idle arm is
# somewhere known instead of wherever the previous run happened to leave it.
IDLE_ZERO_JOINTS = [0.0] * 7


def preview(frames, arms, q_start, conn, cfg, tools=None,
            solver='analytic', diff_ik_config=None):
    """Dry-simulate the whole episode from the measured start pose.

    Exists because a passing dry-run is NOT evidence that the motion is safe.
    The gate checks joint limits, joint rate and tracking error; there is no
    self-collision model anywhere in this stack. Relative mapping preserves
    the demonstrator's *displacement*, not their *clearance* -- a demo whose
    hands sweep 153mm inward from a wide start does the same 153mm inward
    sweep from wherever the robot happens to begin, which may be 20mm from
    its own body. That is exactly how the 2026-08-25 near-collision happened.

    Uses real ArmChannel objects, never sending: a preview built from a
    reimplemented pipeline could disagree with the pipeline that then runs.

    Both the fingertip and the flange are tracked. The keep-out box guards
    the robot's own body, and with a tool frame set those are two different
    points several centimetres apart -- the tool is what gets there first,
    but the flange is what the 100mm floor was originally read off.

    solver/diff_ik_config MUST be whatever the live run will use: a
    preview solved by a different solver is a preview of a different
    trajectory, and diff-IK and the analytic solver pick different
    elbow configurations for the same target (docs/algos.md#diff_ikpy).

    :return: {arm: {'lo'/'hi': tip envelope, 'flo'/'fhi': flange envelope,
                    'start': [...], 'ok': n, 'blocked': n,
                    'max_rate': deg/s, 'tool': ToolFrame|None}}
    """
    period = 1.0 / cfg.CONTROL_HZ
    tools = tools or {}
    sim = {a: ArmChannel(a, HAND_OF_ARM[a], conn, cfg, solver=solver,
                         diff_ik_config=diff_ik_config,
                         axis_map=None, tool=tools.get(a),
                         delta_frame='ee',
                         quiet=True)     # the live channels already warned
           for a in arms}
    out = {}
    for a in arms:
        ch = sim[a]
        ch.q_cmd = list(q_start[a])
        p0 = ch.tip(ch.q_cmd)
        f0 = list(ch.ik.fk_xyzabc(ch.q_cmd)[:3])
        out[a] = {'start': p0, 'lo': list(p0), 'hi': list(p0),
                  'flo': list(f0), 'fhi': list(f0),
                  'ok': 0, 'blocked': 0, 'max_rate': 0.0,
                  'tool': tools.get(a)}
    for f in frames:
        for a in arms:
            ch, st = sim[a], out[a]
            prev = list(ch.q_cmd)
            # q_meas = q_cmd: perfect servo tracking, so the preview measures
            # the trajectory rather than the (absent) servo lag.
            q = ch.step(f, 0.0, period, list(ch.q_cmd))
            if q is None:
                st['blocked'] += 1
                continue
            st['ok'] += 1
            st['max_rate'] = max(st['max_rate'],
                                 max(abs(x - y) for x, y in zip(q, prev)) / period)
            pos = ch.tip(q)
            flange = ch.ik.fk_xyzabc(q)[:3]
            for k in range(3):
                st['lo'][k] = min(st['lo'][k], pos[k])
                st['hi'][k] = max(st['hi'][k], pos[k])
                st['flo'][k] = min(st['flo'][k], flange[k])
                st['fhi'][k] = max(st['fhi'][k], flange[k])
    return out


def report_preview(pv, cfg):
    """Print the envelope and check it against the keep-out box.

    Returns False if any arm violates config.UMI_REPLAY_KEEPOUT.
    """
    keep = getattr(cfg, 'UMI_REPLAY_KEEPOUT', None) or {}
    ok = True
    print('\n=== 下发前预演（未发送任何指令）===')
    for arm, st in sorted(pv.items()):
        total = st['ok'] + st['blocked']
        what = '指尖' if st.get('tool') else 'TCP(法兰)'
        print('臂%s  起点 %s [%+7.1f %+7.1f %+7.1f] mm   跟随 %d/%d 帧   '
              '峰值关节速度 %.1f°/s (额度 %.0f)'
              % (arm, what, st['start'][0], st['start'][1], st['start'][2],
                 st['ok'], total, st['max_rate'], cfg.MAX_JOINT_RATE_DEG_S))
        for k, ax in enumerate('XYZ'):
            span = st['hi'][k] - st['lo'][k]
            print('      %s  %+7.1f .. %+7.1f mm   (行程 %5.1f mm)'
                  % (ax, st['lo'][k], st['hi'][k], span))
        if st['blocked']:
            print('      ⚠️ %d 帧未通过 IK/安全闸 —— 起始构型可能不合适'
                  % st['blocked'])
            ok = False
        box = keep.get(arm)
        if box:
            # Check the fingertip AND the flange: with a tool frame set they
            # are tens of mm apart, and the box is about the body, not about
            # whichever point happens to be convenient.
            for label, lo_k, hi_k in (('指尖', 'lo', 'hi'),
                                      ('法兰', 'flo', 'fhi')):
                if label == '指尖' and not st.get('tool'):
                    continue        # same point, don't print it twice
                for k, ax in enumerate('XYZ'):
                    lim = box.get(ax)
                    if not lim:
                        continue
                    if st[lo_k][k] < lim[0] or st[hi_k][k] > lim[1]:
                        print('      ❌ %s %s 轴越界：轨迹 %+.1f..%+.1f，'
                              '允许 %+.1f..%+.1f'
                              % (label, ax, st[lo_k][k], st[hi_k][k],
                                 lim[0], lim[1]))
                        ok = False
        else:
            print('      （臂%s 未配置 UMI_REPLAY_KEEPOUT，只能靠你看数字判断）' % arm)
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='UMI/LeRobot v3 episode directory')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--frames', default=None, metavar='A:B',
                    help='Replay only this half-open range of RECORDED '
                         'frames -- the spans scripts/umi_filter.py prints '
                         'after cutting what this robot cannot do.')
    ap.add_argument('--tool', default='umi',
                    help="configs/tool/<name>.yaml to take the "
                         "flange->fingertip offset from, or 'none' to "
                         'retarget the flange (pre-tool_frame.py behaviour).')
    ap.add_argument('--tip', default=None,
                    help='Override the tool offset: x,y,z[,a,b,c] mm/deg.')
    ap.add_argument('--arms', default=None,
                    help='Defaults to whichever hands the episode contains. '
                         'Selecting a subset also parks every other arm at '
                         'all-zero joints before the run -- see --no-idle-zero.')
    ap.add_argument('--no-idle-zero', action='store_true',
                    help='Do NOT park the arms excluded by --arms at all-zero '
                         'joints first; leave them wherever they are. Only '
                         'meaningful with a partial --arms.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Connect and run the full pipeline, send nothing.')
    ap.add_argument('--gripper', action='store_true',
                    help='Also drive the UMI follower grippers.')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='Motion scale; 1.0 (default) reproduces the demo.')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='Playback rate. 0.3 = slow rehearsal, same path.')
    ap.add_argument('--vel-ratio', type=int, default=None)
    ap.add_argument('--solver', choices=['analytic', 'diff'], default='diff',
                    help='IK solver. Defaults to diff (algos/diff_ik.py) on '
                         'this path, unlike run_teleop.py -- see '
                         'docs/scripts.md#umi_replaypy. analytic '
                         '(ik_solver.ArmIK) is still selectable.')
    ap.add_argument('--solver-config', default=None,
                    help='--solver diff only: YAML path '
                         '(default configs/solver/diff.yaml).')
    ap.add_argument('--goto-start', action='store_true',
                    help='Move to config.UMI_START_JOINTS first, then replay. '
                         "The equivalent of UMI's --init_joints; without it "
                         'the replay starts from wherever the arms already are.')
    ap.add_argument('--yes', action='store_true',
                    help='Skip the pre-flight confirmation prompt.')
    ap.add_argument('--no-preview', action='store_true',
                    help='Skip the pre-flight simulation. Not recommended -- '
                         'it is the only thing in this stack that can show a '
                         'trajectory heading into the robot body.')
    ap.add_argument('--no-home', action='store_true',
                    help='Skip parking on exit (leaves the arms where they stop).')
    ap.add_argument('--park', default='start', choices=['start', 'home'],
                    help="Where to park on exit. 'start' (default) = "
                         'config.UMI_START_JOINTS, so a rerun can go straight '
                         'into --dry-run/replay without --goto-start; '
                         "'home' = config.HOME_JOINTS, which this trajectory "
                         'does NOT have clearance to start from.')
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--viewer-endpoint', default='tcp://127.0.0.1:5568')
    args = ap.parse_args()

    if args.speed <= 0:
        print('❌ --speed 必须为正'); return 2
    config.SCALE = args.scale
    if args.vel_ratio is not None:
        config.apply_vel_ratio(args.vel_ratio)

    solver_cfg = None
    if args.solver == 'diff':
        try:
            solver_cfg, solver_cfg_path = load_diff_ik_config(
                args.solver_config, _REPO_ROOT)
        except ValueError as e:
            print('❌ %s' % e)
            return 2
        print('求解器: diff-IK  配置 %s\n%s'
              % (solver_cfg_path, solver_cfg.model_dump()))
    else:
        print('求解器: analytic (ik_solver.ArmIK)')

    if not args.dry_run and not config.CALIBRATED:
        print('❌ config.CALIBRATED=False —— 坐标系还没标定，拒绝真实下发。\n'
              '   先用 replay.py --umi 离线确认方向，再上机。')
        return 2

    # ---- source ----
    raw, info = load_episode(args.dataset, args.episode)
    hands = [h for h in HANDS if raw[0].pose(h) is not None]
    print('载入 %d 帧 @ %sHz ← %s (episode %d, %s)  有效手: %s'
          % (len(raw), info.get('fps'), args.dataset, args.episode,
             info.get('robot_type'), hands))
    if args.frames:
        raw = slice_frames(raw, args.frames)
        if len(raw) < 2:
            print('❌ --frames %s 只剩 %d 帧，没法回放' % (args.frames, len(raw)))
            return 2
        print('  截取 --frames %s -> %d 帧（录制帧下标）'
              % (args.frames, len(raw)))

    arms = list(args.arms.upper()) if args.arms \
        else [config.ARM_OF_HAND[h] for h in hands]
    for a in arms:
        if a not in config.ARM_OF_HAND.values():
            print('❌ 未知的臂 %r' % a); return 2
        if HAND_OF_ARM[a] not in hands:
            print('❌ 臂 %s 需要 %s 手的数据，episode 里没有'
                  % (a, HAND_OF_ARM[a]))
            return 2

    # A recorded UMI pose is the leader gripper's FINGERTIP MIDPOINT; FK here
    # returns the flange. Following one at the other is invisible in pure
    # translation and wrong the moment the wrist turns -- see
    # docs/algos.md#tool_framepy.
    tools = {}
    for a in arms:
        if args.tip:
            tools[a] = ToolFrame([float(v) for v in args.tip.split(',')],
                                 label='--tip')
        elif args.tool.lower() in ('none', 'no', 'flange'):
            tools[a] = None
        else:
            tools[a], tool_path = ToolFrame.load(args.tool, a, _REPO_ROOT)
            if tools[a] is None:
                print('⚠️  臂%s 指尖偏移未标定（%s 的 kine_offset 全零）——'
                      '按法兰跟随，腕部旋转时指尖会走出 demo 里没有的弧线。'
                      '先跑 scripts/tip_calib.py。' % (a, tool_path))
        if tools[a] is not None:
            print('臂%s 指尖偏移 %s' % (a, tools[a]))

    # Divide, don't multiply: resampling at CONTROL_HZ/speed and then
    # playing back at CONTROL_HZ is what makes --speed a pure time rescale.
    # speed=0.5 asks for twice as many interpolated frames, which take twice
    # as long to emit -- same geometric path, half the Cartesian velocity, so
    # a slow rehearsal exercises the identical poses under a looser budget.
    frames = resample(raw, config.CONTROL_HZ / args.speed)
    print('插值到 %d 个控制周期 @ %.0fHz  ->  回放 %.1f 秒（%.2fx 速度）'
          % (len(frames), config.CONTROL_HZ,
             len(frames) / config.CONTROL_HZ, args.speed))
    config.summary()
    if args.scale != 1.0:
        print('⚠️  scale=%.2f，回放的是缩放后的轨迹，不是原始 demo' % args.scale)
    if args.speed > 1.0:
        print('⚠️  speed=%.2f 比原始 demo 更快，笛卡尔速度按同比例上升 —— '
              '这段 demo 在 1.0x 就已经贴着速度预算了' % args.speed)
    print('模式: %s' % ('dry-run（不下发）' if args.dry_run else '⚠️ 真实下发'))
    print()

    # ---- hardware ----
    print('连接机器人 %s…' % config.ROBOT_IP)
    conn = RobotConnection(config.ROBOT_IP, dry_run=args.dry_run)
    print('✅ 控制器版本 %s' % conn.version)
    # ensure_clear, not check_and_clear_errors: the latter fires clear_error()
    # once with no confirmation, which its own docstring notes is not enough
    # right after an e-stop -- err=13 (Emcy) stays latched even once the
    # physical button is released. Hit exactly this on 2026-08-25.
    conn.check_and_clear_errors()

    viewer = ViewerBridge(args.viewer, args.viewer_endpoint)
    if args.viewer:
        print('✅ 查看器镜像已开启 -> %s' % args.viewer_endpoint)

    channels = {a: ArmChannel(a, HAND_OF_ARM[a], conn, config,
                              solver=args.solver, diff_ik_config=solver_cfg,
                              axis_map=None, tool=tools[a],
                              delta_frame='ee')
                for a in arms}
    # Arms present on this robot that this run is not replaying. They still
    # need a driver: to be zeroed below, and to be disabled on exit.
    idle_arms = ([] if (args.no_idle_zero or not args.arms)
                 else [a for a in config.ARMS if a not in arms])
    idle_drivers = {a: ArmDriver(conn, a, config) for a in idle_arms}
    if idle_arms:
        print('未回放的臂: %s —— 开跑前先归零到全零关节'
              % '/'.join(idle_arms))
        if not args.goto_start:
            print('⚠️  没有 --goto-start：回放臂会从当前构型起步，'
                  '不是 config.UMI_START_JOINTS')

    grippers = None
    try:
        for a, d in idle_drivers.items():
            ok, st = ensure_clear(conn, d, a)
            if not ok:
                print('❌ 臂%s（待归零）清错失败（cur=%s err=%s）'
                      % (a, st['cur'], st['err']))
                return 2
            d.prepare()
        for a, ch in channels.items():
            ok, st = ensure_clear(conn, ch.driver, a)
            if not ok:
                print('❌ 臂%s 清错失败（cur=%s err=%s）。若刚拍过急停，'
                      '先确认物理急停按钮已复位再重试。' % (a, st['cur'], st['err']))
                return 2
        for ch in channels.values():
            ch.prepare()

        # ---- park the arms this run is NOT replaying ----
        # A single-arm run of a bimanual demo leaves the other arm wherever
        # the last run stopped, which is not a state anybody reasoned about.
        # Zero it first, before the replayed arm moves into the workspace.
        if idle_arms and not args.no_idle_zero:
            if args.dry_run:
                print('\n[dry-run] 跳过臂 %s 归零（全零关节）'
                      % '/'.join(idle_arms))
            else:
                print('\n=== 未回放的臂先归零（全零关节）===')
                for a in idle_arms:
                    print('  %s -> %s  （当前 %s）'
                          % (a, IDLE_ZERO_JOINTS,
                             [round(v, 2) for v in idle_drivers[a].joints()]))
                if not args.yes:
                    try:
                        if input('回车确认，其它键取消：').strip():
                            print('已取消'); return 0
                    except EOFError:
                        print('❌ 非交互环境且未加 --yes，取消'); return 2
                ok, why, _ = move_to_joints(
                    conn, idle_drivers,
                    {a: list(IDLE_ZERO_JOINTS) for a in idle_arms},
                    speed_deg_s=config.HOME_SPEED_DEG_S, hz=config.CONTROL_HZ,
                    max_err_deg=config.MAX_TRACKING_ERR_DEG)
                if not ok:
                    print('❌ 臂 %s 归零未到位（%s），中止'
                          % ('/'.join(idle_arms), why))
                    return 2
                print('✅ 臂 %s 已归零' % '/'.join(idle_arms))

        # ---- optional: move to the start pose first ----
        if args.goto_start and not args.dry_run:
            targets = {}
            for a in arms:
                if a not in config.UMI_START_JOINTS:
                    print('❌ config.UMI_START_JOINTS 里没有臂%s' % a); return 2
                targets[a] = list(config.UMI_START_JOINTS[a])
            print('\n=== 移动到 UMI 起始构型 ===')
            for a in arms:
                print('  %s -> %s' % (a, targets[a]))
            if not args.yes:
                try:
                    if input('回车确认，其它键取消：').strip():
                        print('已取消'); return 0
                except EOFError:
                    print('❌ 非交互环境且未加 --yes，取消'); return 2
            ok, why, _ = move_to_joints(
                conn, {a: channels[a].driver for a in arms}, targets,
                speed_deg_s=config.UMI_START_SPEED_DEG_S, hz=config.CONTROL_HZ,
                max_err_deg=config.MAX_TRACKING_ERR_DEG)
            if not ok:
                print('❌ 未到位（%s），中止' % why); return 2
            for a, ch in channels.items():
                ch.q_cmd = list(ch.driver.joints())
            print('✅ 已到位')

        # ---- pre-flight: simulate before anything moves ----
        if not args.dry_run and not args.no_preview:
            q_start = {a: list(ch.q_cmd) for a, ch in channels.items()}
            print('\n预演中（%d 帧 × %d 臂）…' % (len(frames), len(arms)))
            pv = preview(frames, arms, q_start, conn, config, tools,
                         solver=args.solver, diff_ik_config=solver_cfg)
            safe = report_preview(pv, config)
            print('\n本机 A 臂基座系: X 前 / Y 下 / Z 左（由 config.AXIS_MAP 推出）——'
                  '\nZ 越小越靠近本体中心线，这正是 2026-08-25 那次险些撞机的方向。')
            if not safe:
                print('\n❌ 预演不通过，拒绝下发。先用 scripts/goto_joints.py 换一个'
                      '起始构型（给扫掠方向留出余量），或降低 --scale。')
                return 2
            if not args.yes:
                try:
                    ans = input('\n以上包络是否可接受？手放在急停上。输入 yes 继续：')
                except EOFError:
                    print('❌ 非交互环境且未加 --yes，拒绝下发。')
                    return 2
                if ans.strip().lower() not in ('yes', 'y'):
                    print('已取消，未下发任何指令。')
                    return 0

        if args.gripper:
            from drivers.umi_gripper import UmiGrippers            # noqa: E402
            config.UMI_GRIPPER_ENABLED = True
            grippers = UmiGrippers(config, arms).start()
            print('✅ 夹爪已就绪')

        drivers = {a: c.driver for a, c in channels.items()}
        # ControlLoop only ever *sends* what step() returns, and step() never
        # names an idle arm -- but it disables every driver it is handed, so
        # the zeroed arms belong here or they would be left energized.
        loop = ControlLoop(conn, dict(drivers, **idle_drivers),
                           config.CONTROL_HZ)
        state_i = [0]
        last_report = [time.monotonic()]
        # rt.cart_lag_mm is instantaneous -- reading it after the loop reports
        # the LAST frame's lag (near zero once the trajectory settles), not the
        # worst one. Track the max as it goes.
        max_lag = {a: 0.0 for a in arms}

        def step(dt):
            i = state_i[0]
            if i >= len(frames):
                loop.running = False
                return None
            state_i[0] = i + 1
            frame = frames[i]

            state = conn.subscribe()
            viewer.publish_state(state, step=i, max_steps=len(frames))

            cmds = {}
            for arm, ch in channels.items():
                idx = 0 if arm == 'A' else 1
                q_meas = list(state['outputs'][idx]['fb_joint_pos'])
                if conn.dry_run and ch.q_cmd is not None:
                    # Nothing is sent in dry-run, so the arm stays parked while
                    # q_cmd walks away from it. The resulting "tracking error"
                    # measures the dry-run, not the trajectory, and it trips the
                    # gate within ~0.5s every time. Assume perfect servo
                    # tracking instead -- the same assumption replay.py's
                    # offline loop documents.
                    q_meas = list(ch.q_cmd)
                # age_s=0: a recording is never stale, and letting the gate's
                # XR-staleness check fire here would only ever be a false
                # positive.
                q = ch.step(frame, 0.0, dt, q_meas)
                max_lag[arm] = max(max_lag[arm], ch.rt.cart_lag_mm)
                if q is not None:
                    cmds[arm] = q
                if grippers is not None:
                    x = frame.gripper(ch.hand)
                    if x is not None:
                        grippers.set_target(arm, x)   # 0=closed, no conversion

            # A recording has no clutch to release, so Retargeter's fault
            # latch can never clear (see drivers/umi_source.py) -- the run would
            # otherwise coast to the end doing nothing while still printing a
            # healthy-looking report. Stop instead: a replay that faulted is not
            # a replay that succeeded.
            faulted = [a for a, ch in channels.items() if ch.rt.fault_count]
            if faulted:
                print('\n❌ 臂 %s 触发保护闭锁，录制回放没有离合可松开，'
                      '无法重新进入 —— 中止（第 %d/%d 帧）'
                      % ('/'.join(sorted(faulted)), i, len(frames)))
                loop.running = False
                return None

            if grippers is not None and grippers.check():
                print('\n❌ 夹爪保护触发：%s' % grippers.fault)
                loop.running = False
                return None

            now = time.monotonic()
            if now - last_report[0] >= 2.0:
                last_report[0] = now
                bits = ['%d/%d (%.0f%%)'
                        % (i, len(frames), 100.0 * i / len(frames))]
                for arm, ch in channels.items():
                    bits.append('%s:%s err %.2f°/%.1f°'
                                % (arm, ch.blocked_reason or '跟随中',
                                   ch.track_err, config.MAX_TRACKING_ERR_DEG))
                if grippers is not None:
                    for arm in arms:
                        o = grippers.observation(arm)
                        if o is not None:
                            bits.append('爪%s %.2f/%+.3fN·m'
                                        % (arm, o.position, o.torque))
                print('  ' + '  '.join(bits))
            return cmds

        if args.no_home:
            before_disable = None
        else:
            park_to = (config.UMI_START_JOINTS if args.park == 'start'
                       else config.HOME_JOINTS)
            park_label = 'UMI 起始构型' if args.park == 'start' else '归位'

            def before_disable():
                # Grippers first: they are on their own link, but stopping
                # them before the arms sweep to the parking pose keeps the
                # jaws from holding an object through that motion.
                if grippers is not None:
                    grippers.stop()
                # Park at UMI_START_JOINTS, not HOME: HOME is where this
                # trajectory's inward sweep ends 22mm from the body, so
                # parking there would leave the arms somewhere the next run
                # cannot start from. See umi与天机replay.md.
                home_arms(conn, channels, park_to, label=park_label)

        print('\n▶ 开始回放，Ctrl+C 中止\n')
        stats = loop.run(step, before_disable=before_disable)

        print('\n=== 统计 ===')
        print('回放 %d/%d 帧，循环 %d 次，下发 %d 次，超时 %d 次，最大抖动 %.1f ms'
              % (state_i[0], len(frames), stats['iters'], stats['sent'],
                 stats['overruns'], stats['max_jitter_ms']))
        for arm, ch in channels.items():
            # The diff solver never calls ch.ik.solve*(), so ArmIK's
            # backoff/projection counters stay at zero and would read as
            # "nothing ever had to back off" -- report diff_solver's own
            # stats instead, same as run_teleop.py does.
            if ch.solver == 'diff':
                print('臂%s: 通过安全闸 %d 帧，离合 %d 次，保护闭锁 %d 次，'
                      'diff-IK 统计 %s'
                      % (arm, ch.n_sent, ch.rt.engage_count, ch.rt.fault_count,
                         {k: v for k, v in ch.diff_solver.stats.items() if v}))
                print('     跟踪误差峰值 %.2f° / 上限 %.1f°   '
                      '实际最大帧增量 %.3f° / 额度 %.3f°   钳位 %d 帧   '
                      '单帧求解最大耗时 %.3f ms（%.1f ms 预算内正常）   '
                      '笛卡尔滞后峰值 %.1f mm (限速 %d 帧)'
                      % (ch.track_err_max, config.MAX_TRACKING_ERR_DEG,
                         ch.gate.max_step_used, config.MAX_JOINT_STEP_DEG,
                         ch.gate.stats['clamped'], ch.diff_solve_ms_max,
                         1000.0 / config.CONTROL_HZ, max_lag[arm],
                         ch.rt.n_slewed))
            else:
                print('臂%s: 通过安全闸 %d 帧，离合 %d 次，保护闭锁 %d 次，IK 统计 %s'
                      % (arm, ch.n_sent, ch.rt.engage_count, ch.rt.fault_count,
                         {k: v for k, v in ch.ik.stats.items() if v}))
                print('     跟踪误差峰值 %.2f° / 上限 %.1f°   '
                      '实际最大帧增量 %.3f° / 额度 %.3f°   钳位 %d 帧   '
                      '退让 %d 次   投影 %d 次   笛卡尔滞后峰值 %.1f mm (限速 %d 帧)'
                      % (ch.track_err_max, config.MAX_TRACKING_ERR_DEG,
                         ch.gate.max_step_used, config.MAX_JOINT_STEP_DEG,
                         ch.gate.stats['clamped'], ch.ik.n_backoff,
                         ch.ik.n_projected, max_lag[arm], ch.rt.n_slewed))
        if grippers is not None:
            for arm in arms:
                o = grippers.observation(arm)
                if o is not None:
                    print('爪%s: 终止位置 %.3f  力矩 %+.3f N·m'
                          % (arm, o.position, o.torque))
        incomplete = state_i[0] < len(frames)
        if incomplete:
            print('⚠️  回放未走完（%d/%d 帧）' % (state_i[0], len(frames)))
        return 1 if incomplete else 0
    finally:
        if grippers is not None:
            grippers.stop()             # idempotent; before_disable may have run
            print('夹爪已失能')
        # ControlLoop.run() disables on its own way out, but every early
        # return above (idle-zero missed, goto-start missed, pre-flight
        # rejected) skips it and would leave the arms energized. disable() is
        # idempotent, so call it unconditionally -- "every exit path must
        # call this", per its own docstring in drivers/arm_driver.py.
        # idle_drivers belong here too: they were prepare()d for the zeroing
        # move, so an early return between there and the loop would otherwise
        # leave an arm nobody is watching energized.
        for a, d in list(idle_drivers.items()) + \
                [(a, c.driver) for a, c in channels.items()]:
            try:
                d.disable()
            except Exception as e:
                print('⚠️ 臂%s 下伺服异常：%s' % (a, e))
        viewer.close()
        conn.close()
        print('连接已断开')


if __name__ == '__main__':
    sys.exit(main())
