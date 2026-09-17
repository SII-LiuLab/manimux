#!/usr/bin/env python3
"""Pivot-calibrate the flange -> fingertip offset of the mounted gripper.

Fills in `configs/tool/<tool>.yaml`'s `kine_offset`, which is what
`algos.tool_frame.ToolFrame` needs before a UMI episode can be replayed --
or filtered (`scripts/umi_filter.py`) -- at the point that actually touches
the object. Without it the pipeline follows a recorded *fingertip*
trajectory with the *flange*, which is invisible in pure translation and
wrong the moment the wrist turns. See docs/algos.md#tool_framepy.

READ-ONLY on the robot. It calls `ArmDriver.state()` and nothing else -- no
prepare, no mode switch, no command, ever. Same category as
scripts/get_current_pos.py, and it is safe to run alongside a `set-state
<arm> release` session in another terminal (which is exactly how you are
meant to move the arm for it).

Method
------
Rest the fingertip midpoint on one fixed point, sweep the wrist through as
many orientations as the arm allows, capture the joints each time. Every
sample satisfies `R_i·t + p_i = c`, linear in the unknowns (t, c); see
algos.tool_frame.solve_pivot. It recovers the translation only -- nothing
about a rotationally symmetric point constrains the tool's orientation --
and that is enough, because under relative mapping the tool rotation does
not change a single commanded pose (proved and self-tested in
algos/tool_frame.py). `kine_offset`'s a/b/c stay zero and only matter if you
later hand the same entry to `Marvin_Robot.set_tool()`.

How to touch a midpoint that sits in mid-air: **close the jaws fully first**.
The two pads meet, and symmetric jaws keep the midpoint still as they open,
so a closed-jaw measurement is valid at every opening (that is the same
property xense-taccap-lerobot's ee_transform.py relies on). One ambiguity
this cannot settle: *where along the finger* the recorded TCP sits. The CAD
"EE frame" is the authority there; if the fingers are long, a few mm of
axial disagreement stays in the number.

Usage
-----
    # terminal 1 -- zero-force drag so you can pose the arm by hand
    set-state A release --tool umi

    # terminal 2
    python3 scripts/tip_calib.py --arm A                 # collect + solve
    python3 scripts/tip_calib.py --arm A --solve FILE    # re-solve, no robot
    python3 scripts/tip_calib.py --self-test             # math only

Output: data/tip_calib/<tool>/<arm>/samples.json plus the two YAML lines to
paste into configs/tool/<tool>.yaml.

See docs/scripts.md#tip_calibpy.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                     # noqa: E402
from algos.ik_solver import ArmIK                                 # noqa: E402
from algos.tool_frame import solve_pivot                          # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quality gates. A pivot fit is only as conditioned as the orientation
# spread: with the wrist barely moving, t is unconstrained along the line of
# sight and a tiny residual proves nothing (algos/tool_frame.py's self-test
# demonstrates exactly that failure -- 0.9mm residual, 2.9mm error).
MIN_SAMPLES = 4
MIN_SPREAD_DEG = 60.0
MAX_RESID_MM = 2.0


def collect(arm, ik, out_path):
    """Interactively capture flange poses, one per Enter. Read-only."""
    from drivers.arm_driver import ArmDriver, RobotConnection      # noqa: E402

    print('连接机器人 %s（只读，不下发任何指令）…' % config.ROBOT_IP)
    conn = RobotConnection(config.ROBOT_IP)
    samples = []
    try:
        drv = ArmDriver(conn, arm, config)
        print('✅ 控制器版本 %s\n' % conn.version)
        print('步骤：')
        print('  1. 另一个终端里 `set-state %s release --tool umi`，'
              '让手臂零力可拖' % arm)
        print('  2. 夹爪**完全闭合**')
        print('  3. 把两指中点抵在一个固定的尖点上，别让它挪动')
        print('  4. 保持那个点不动，把手腕摆到尽量不同的姿态，每摆好一次敲回车')
        print('  至少 %d 个点，姿态张角越大越好（≥%.0f° 才算够）；q + 回车结束\n'
              % (MIN_SAMPLES, MIN_SPREAD_DEG))
        while True:
            try:
                s = input('  [%d 个点] 回车采集 / q 结束：' % len(samples))
            except EOFError:
                print('\n（非交互环境，结束采集）')
                break
            if s.strip().lower() in ('q', 'quit', 'exit'):
                break
            st = drv.state()
            if st['err']:
                print('    ⚠️ 臂%s 报错 err=%s，先清错再采' % (arm, st['err']))
                continue
            q = [float(v) for v in st['q']]
            x = ik.fk_xyzabc(q)
            samples.append({'q': q, 'flange_xyzabc': list(x)})
            print('    q = %s' % [round(v, 2) for v in q])
            print('    法兰 [%+.1f %+.1f %+.1f] mm  姿态 [%+.1f %+.1f %+.1f]°'
                  % tuple(x))
    finally:
        conn.close()
        print('连接已断开')

    if samples:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as fh:
            json.dump({'arm': arm, 'robot_ip': config.ROBOT_IP,
                       'kine_cfg': config.KINE_CFG,
                       'collected': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'samples': samples}, fh, indent=2)
        print('已保存 %d 个采样 -> %s' % (len(samples), out_path))
    return samples


def report(samples, arm, tool, src_path):
    poses = [s['flange_xyzabc'] for s in samples]
    if len(poses) < 3:
        print('❌ 只有 %d 个点，解不出来（至少 3，建议 %d 以上）'
              % (len(poses), MIN_SAMPLES))
        return None
    t, c, resid, spread = solve_pivot(poses)

    print('\n=== 标定结果（臂%s，%d 个点）===' % (arm, len(poses)))
    print('法兰 -> 指尖  [%+8.2f %+8.2f %+8.2f] mm   离法兰 %.1f mm'
          % (t[0], t[1], t[2], sum(v * v for v in t) ** 0.5))
    print('触点（基座系）[%+8.2f %+8.2f %+8.2f] mm' % tuple(c))
    print('残差 %.2f mm   姿态张角 %.0f°' % (resid, spread))

    ok = True
    if len(poses) < MIN_SAMPLES:
        print('⚠️ 点数 %d < %d，勉强能解但没有冗余，出错也看不出来'
              % (len(poses), MIN_SAMPLES))
        ok = False
    if spread < MIN_SPREAD_DEG:
        print('❌ 姿态张角只有 %.0f°（要 ≥%.0f）。**残差小在这里不算数** —— '
              '姿态不够散时 t 沿视线方向根本没有被约束，'
              '拟合会给出一个残差很小的错误答案。'
              % (spread, MIN_SPREAD_DEG))
        ok = False
    if resid > MAX_RESID_MM:
        print('❌ 残差 %.2f mm > %.1f mm：多半是采样过程中触点被蹭动了，'
              '或者某个点没真正抵住。重采。' % (resid, MAX_RESID_MM))
        ok = False
    if not ok:
        print('\n→ 没通过质量门槛，先别往 yaml 里填。')
        return None

    print('\n把下面两行填进 configs/tool/%s.yaml 的 arms.%s：' % (tool, arm))
    print('-' * 68)
    print('    kine_offset: [%.3f, %.3f, %.3f, 0.0, 0.0, 0.0]'
          % (t[0], t[1], t[2]))
    print('    kine_offset_source: >-')
    print('      scripts/tip_calib.py 枢轴标定，臂%s，%d 个姿态，张角 %.0f°，'
          % (arm, len(poses), spread))
    rel = os.path.relpath(src_path, _REPO_ROOT)
    print('      残差 %.2f mm，%s。采样数据 %s。'
          % (resid, time.strftime('%Y-%m-%d'),
             src_path if rel.startswith('..') else rel))
    print('      仅平移由标定得到（枢轴法解不出工具姿态，相对映射下也不需要，'
          '见 algos/tool_frame.py）。')
    print('-' * 68)
    print('\n填完用离线回放确认它确实改变了轨迹（不是被忽略了）：')
    print('  python3 replay.py --umi ~/Downloads/replay_test --hand %s'
          % ('left' if arm == 'A' else 'right'))
    print('  python3 scripts/umi_filter.py ~/Downloads/replay_test --arms %s'
          % arm)
    return t


def self_test():
    """The quality gate that matters: a small orientation spread must be
    rejected even though its residual looks great."""
    import math

    import numpy as np

    from algos.retarget import mat_to_xyzabc

    rng = np.random.default_rng(1)
    truth = np.array([9.0, -3.0, 176.0])
    point = np.array([480.0, 100.0, 260.0])

    def poses_with_spread(deg, n=6, noise=0.0):
        out = []
        for _ in range(n):
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            a = math.radians(rng.uniform(-deg, deg))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
            R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
            out.append(mat_to_xyzabc(R, point - R @ truth
                                     + rng.normal(scale=noise, size=3)))
        return out

    t, _, resid, spread = solve_pivot(poses_with_spread(90))
    err = float(np.linalg.norm(np.array(t) - truth))
    print('张角 %.0f°：误差 %.2e mm 残差 %.2e mm  %s'
          % (spread, err, resid, '✅' if err < 1e-6 else '❌'))

    t, _, resid, spread = solve_pivot(poses_with_spread(8, noise=0.3))
    err = float(np.linalg.norm(np.array(t) - truth))
    bad = spread < MIN_SPREAD_DEG
    print('张角 %.0f°：误差 %.1f mm，但残差只有 %.2f mm —— 门槛拦下: %s'
          % (spread, err, resid, '✅' if bad else '❌ 没拦住'))
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--tool', default='umi',
                    help='which configs/tool/<name>.yaml this is for '
                         '(only decides where samples are stored and what '
                         'the pasteable snippet says).')
    ap.add_argument('--solve', metavar='FILE', default=None,
                    help='Re-solve from a saved samples.json instead of '
                         'collecting. No robot needed.')
    ap.add_argument('--self-test', action='store_true',
                    help='Math only: no robot, no dataset.')
    args = ap.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    if args.solve:
        with open(args.solve) as fh:
            doc = json.load(fh)
        arm = doc.get('arm', args.arm)
        return 0 if report(doc['samples'], arm, args.tool, args.solve) else 1

    out_path = os.path.join(_REPO_ROOT, 'data', 'tip_calib', args.tool,
                            args.arm, 'samples.json')
    ik = ArmIK(arm_type=0 if args.arm == 'A' else 1,
               config_path=config.KINE_CFG)
    samples = collect(args.arm, ik, out_path)
    if not samples:
        print('没有采到点，什么也没算。')
        return 1
    return 0 if report(samples, args.arm, args.tool, out_path) else 1


if __name__ == '__main__':
    sys.exit(main())
