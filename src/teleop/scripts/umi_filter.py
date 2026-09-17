#!/usr/bin/env python3
"""Embodiment feasibility filter for UMI episodes — cut the parts this robot
cannot do, keep the parts it can. No robot, no network, nothing sent.

The hand-held rig has no joint limits, no velocity budget and a wrist that
goes where a 7-DoF arm cannot follow. This walks a recorded episode through
the real pipeline (retarget → IK → safety gate, the same objects
`core.arm_channel.ArmChannel` builds) and labels every control period, then
emits the spans worth keeping — as recorded frame indices, which is what a
dataset consumer indexes by.

    python3 scripts/umi_filter.py ~/Downloads/replay_test
    python3 scripts/umi_filter.py ~/Downloads/replay_test --arms A --speed 0.5
    python3 scripts/umi_filter.py ~/Downloads/replay_test --out mask.json
    python3 scripts/umi_filter.py ~/Downloads/replay_test --sweep

Then replay a kept span to see it for yourself:

    python3 replay.py --umi ~/Downloads/replay_test --hand left --frames 0:144
    python3 scripts/umi_replay.py ~/Downloads/replay_test --frames 0:144 \
        --goto-start --arms A --speed 0.3

Read `--speed` carefully. It is the replay speed the verdicts are FOR: the
speed budget scales with it, the workspace does not. A span cut as `speed` at
1.0x may be perfectly feasible at 0.5x — `--sweep` answers exactly that
question — while a span cut as `ik:*` is out of reach at any speed.

⚠️ Fingertip alignment. A recorded UMI pose is the *leader gripper's
fingertip midpoint*, and this stack's FK returns the *flange*. The offset
lives in `configs/tool/<tool>.yaml`'s `kine_offset` (measure it with
scripts/tip_calib.py). While it is all-zero the filter judges a flange
trajectory instead of a fingertip one and says so on every run — see
docs/algos.md#tool_framepy.

See docs/scripts.md#umi_filterpy.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                     # noqa: E402
from algos import embodiment as emb                               # noqa: E402
from algos.tool_frame import ToolFrame                            # noqa: E402
from drivers.umi_source import HANDS, load_episode, resample      # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_OF_ARM = {v: k for k, v in config.ARM_OF_HAND.items()}


def resolve_tool(args, arm):
    """(ToolFrame or None, human-readable provenance)."""
    if args.tip:
        vals = [float(v) for v in args.tip.replace(' ', '').split(',')]
        return ToolFrame(vals, label='--tip'), '--tip %s' % args.tip
    if args.tool.lower() in ('none', 'no', 'flange'):
        return None, '--tool none（按法兰判定）'
    tool, path = ToolFrame.load(args.tool, arm, _REPO_ROOT)
    if tool is None:
        return None, '%s 的 kine_offset 全零 = 未标定' % path
    return tool, '%s kine_offset' % path


def resolve_start(args, arm):
    if args.start_joints:
        q = [float(v) for v in args.start_joints.replace(' ', '').split(',')]
        if len(q) != 7:
            raise SystemExit('❌ --start-joints 需要 7 个关节角，给了 %d 个' % len(q))
        return q, '--start-joints'
    name = 'HOME_JOINTS' if args.start == 'home' else 'UMI_START_JOINTS'
    table = getattr(config, name)
    if arm not in table:
        raise SystemExit('❌ config.%s 里没有臂 %s' % (name, arm))
    return list(table[arm]), 'config.%s' % name


def run_arm(frames, raw, arm, cfg, args, tool, q0, ik):
    """Scan + segment one arm."""
    scan = emb.scan_arm(frames, HAND_OF_ARM[arm], arm, cfg, q0, tool=tool,
                        keepout=(cfg.UMI_REPLAY_KEEPOUT.get(arm)
                                 if args.keepout else None),
                        max_lag_mm=args.max_lag_mm,
                        max_rot_lag_deg=args.max_rot_lag_deg,
                        resync=not args.no_resync, ik=ik)
    min_len = max(1, int(round(args.min_seg_s * cfg.CONTROL_HZ)))
    bridge = max(0, int(round(args.bridge_s * cfg.CONTROL_HZ)))
    scan.segments = emb.stamp_times(
        emb.segment(scan.verdicts, min_len, bridge), frames, raw)
    return scan


def recheck(scan, frames, arm, cfg, args, tool, q0, ik):
    """Re-run each kept span on its own, from q0.

    The main scan is one continuous take: segment 3 starts wherever segment 2
    left the arm. That is the honest model for replaying the episode straight
    through, and the wrong one for using a span on its own -- then the arm
    goes to q0 first, and 'a span that only worked because the arm happened to
    be somewhere convenient' is a span that will not replay. Cheap enough to
    just do (one extra IK pass per span).
    """
    for seg in scan.segments:
        if not seg.ok:
            continue
        sub = emb.scan_arm(frames[seg.start:seg.end], HAND_OF_ARM[arm], arm,
                           cfg, q0, tool=tool,
                           keepout=(cfg.UMI_REPLAY_KEEPOUT.get(arm)
                                    if args.keepout else None),
                           max_lag_mm=args.max_lag_mm,
                           max_rot_lag_deg=args.max_rot_lag_deg,
                           resync=False, ik=ik)
        n_ok = sub.n_feasible
        seg.recheck = {'ok': n_ok, 'n': sub.n,
                       'pass': n_ok == sub.n,
                       'reasons': {k: v for k, v in sub.counts().items()
                                   if k not in emb.FEASIBLE}}
        if not seg.recheck['pass']:
            seg.ok = False
            seg.reasons['recheck_failed'] = sub.n - n_ok


def spans(scan, ok=True):
    return [(s.start, s.end) for s in scan.segments if s.ok == ok]


def fmt_seg(seg, cfg):
    bits = ['帧 %5d..%-5d' % (seg.start, seg.end),
            '录制 %4s..%-4s' % (seg.rec_start, seg.rec_end),
            '%5.2f..%5.2fs' % (seg.t_start, seg.t_end),
            '%5.2f 秒' % (seg.n / cfg.CONTROL_HZ)]
    return '  '.join(bits)


def report(scan, cfg, args):
    print('\n=== 臂%s（%s 手）===' % (scan.arm, scan.hand))
    c = scan.counts()
    n = max(scan.n, 1)
    print('逐帧可跟随 : %d/%d = %.2f%%   (ok %d / slow %d)'
          % (scan.n_feasible, scan.n, 100.0 * scan.n_feasible / n,
             c.get(emb.OK, 0), c.get(emb.SLOW, 0)))
    bad = {k: v for k, v in c.items() if k not in emb.FEASIBLE}
    if bad:
        print('不可跟随   : %d 帧  %s' % (scan.n - scan.n_feasible,
                                          dict(sorted(bad.items()))))
    print('最近限位   : %s   峰值关节速率 %.2f× 额度(%.0f°/s)   '
          '笛卡尔滞后峰值 %.1f mm / %.1f°'
          % ('%.1f°' % scan.worst_margin if scan.worst_margin is not None
             else 'n/a', scan.max_rate_ratio, cfg.MAX_JOINT_RATE_DEG_S,
             scan.max_lag_mm, scan.max_rot_lag_deg))
    if scan.breaks:
        print('重新锚定   : %d 次（第 %s 帧）—— 每一次都意味着一段被切断'
              % (len(scan.breaks), scan.breaks[:8]
                 if len(scan.breaks) <= 8 else
                 '%s…' % scan.breaks[:8]))
    if scan.ik_stats:
        print('IK 统计    : %s' % scan.ik_stats)

    keep = [s for s in scan.segments if s.ok]
    drop = [s for s in scan.segments if not s.ok]
    kept_n = sum(s.n for s in keep)
    print('保留 %d 段 / %d 帧 = %.2f 秒（%.1f%% 的 episode）'
          % (len(keep), kept_n, kept_n / cfg.CONTROL_HZ,
             100.0 * kept_n / n))
    for i, s in enumerate(keep, 1):
        extra = ''
        if s.bridged:
            extra += '  桥接 %d 帧' % s.bridged
        if s.max_rate_ratio > 1.0:
            extra += '  速率峰值 %.2f×' % s.max_rate_ratio
        if s.recheck is not None:
            extra += '  起点复检 %s' % ('✅' if s.recheck['pass'] else '❌')
        print('  ✅ #%d %s%s' % (i, fmt_seg(s, cfg), extra))
        if args.verbose and s.lo:
            for k, ax in enumerate('XYZ'):
                print('        %s %+8.1f .. %+8.1f mm  (行程 %6.1f)'
                      % (ax, s.lo[k], s.hi[k], s.hi[k] - s.lo[k]))
    for s in drop:
        print('  ❌ 剔除 %s  %s' % (fmt_seg(s, cfg),
                                    dict(sorted(s.reasons.items()))))
    if not keep:
        print('  （没有任何一段可用 —— 先看下面的建议，别急着改数据）')


def advise(scans, cfg, args):
    print('\n=== 建议 ===')
    worst_rate = max((s.max_rate_ratio for s in scans), default=0.0)
    reasons = {}
    for s in scans:
        for code, cnt in s.counts().items():
            if code not in emb.FEASIBLE:
                reasons[code] = reasons.get(code, 0) + cnt
    if not reasons:
        n_slow = sum(s.counts().get(emb.SLOW, 0) for s in scans)
        print('· 没有需要剔除的帧%s。'
              % ('（%d 帧被限速钳位，但滞后仍在 %.0fmm 以内，算跟得住）'
                 % (n_slow, args.max_lag_mm) if n_slow else ''))
    # The rate ratio saturates: ArmIK rejects any solution stepping more than
    # IK_MAX_STEP_DEG in one period (branch-jump detection), so no accepted
    # frame can ever report more than that ceiling. At the ceiling the number
    # says "at least this fast", and 1/ratio is not a usable speed suggestion.
    ceiling = cfg.IK_MAX_STEP_DEG * cfg.CONTROL_HZ / cfg.MAX_JOINT_RATE_DEG_S
    if worst_rate >= 0.98 * ceiling:
        print('· 峰值关节速率顶到了 %.2f× —— 这是 IK 分支跳变阈值'
              '（IK_MAX_STEP_DEG=%.1f°/周期）给出的**上限，不是真实需求**，'
              '实际需要多快看不出来。用 --sweep 实测哪个速度能跟上，'
              '别拿 1/比值 当建议。' % (worst_rate, cfg.IK_MAX_STEP_DEG))
    elif worst_rate > 1.0:
        print('· 峰值关节速率是额度的 %.2f 倍 —— 单纯的速度问题，'
              '用 --speed %.2f 重跑（或按 docs/config.md#speed-budget 小步'
              '提 --vel-ratio）就能消掉，而不是把这些帧丢掉。'
              % (worst_rate, min(1.0, 1.0 / worst_rate)))
    if any(k.startswith('ik:') or k.startswith('gate:') for k in reasons):
        print('· 有 IK/限位类拒解：那是构型问题，降速没用。先换起始构型'
              '（--start-joints 试别的），再考虑 --scale 缩小运动。')
    if any(k.startswith('keepout') for k in reasons):
        print('· 有 keepout 越界：config.UMI_REPLAY_KEEPOUT 是**临时值不是实测**'
              '（见 config.py 注释），确认过本体包络再决定是真剔还是改盒子。')
    if args.tool_note_unset:
        print('· 指尖偏移仍未标定，上面每一条结论判的都是法兰轨迹。'
              '先跑 scripts/tip_calib.py 把 configs/tool/umi.yaml 的 '
              'kine_offset 填上，再重跑这个过滤器。')


def sweep(frames_raw, raw, arms, cfg, args, tools, starts, iks):
    print('\n=== 速度扫描（同一条轨迹，只改回放速度）===')
    print('%-6s %s' % ('速度', '  '.join('臂%s 可跟随 / 峰值速率' % a
                                          for a in arms)))
    for sp in args.sweep:
        cells = []
        for arm in arms:
            fr = resample(raw, cfg.CONTROL_HZ / sp)
            sc = emb.scan_arm(fr, HAND_OF_ARM[arm], arm, cfg, starts[arm],
                              tool=tools[arm],
                              keepout=(cfg.UMI_REPLAY_KEEPOUT.get(arm)
                                       if args.keepout else None),
                              max_lag_mm=args.max_lag_mm,
                              max_rot_lag_deg=args.max_rot_lag_deg,
                              resync=not args.no_resync, ik=iks[arm])
            cells.append('%6.2f%%  %.2f×'
                         % (100.0 * sc.n_feasible / max(sc.n, 1),
                            sc.max_rate_ratio))
        print('%-6.2f %s' % (sp, '  '.join(cells)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='UMI/LeRobot v3 episode directory')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--arms', default=None,
                    help='Defaults to whichever hands the episode contains.')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='Motion scale, as in umi_replay.py. 1.0 = the demo.')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='Replay speed the verdicts are for. 1.0 = original.')
    ap.add_argument('--vel-ratio', type=int, default=None)
    ap.add_argument('--start', default='umi', choices=['umi', 'home'],
                    help='Start configuration: config.UMI_START_JOINTS '
                         '(default) or config.HOME_JOINTS.')
    ap.add_argument('--start-joints', default=None,
                    help='7 comma-separated joint angles, overriding --start '
                         'for every scanned arm (e.g. the pose you just read '
                         'with get-current-pos).')
    ap.add_argument('--tool', default='umi',
                    help="configs/tool/<name>.yaml to take the fingertip "
                         "offset from, or 'none' to judge the flange.")
    ap.add_argument('--tip', default=None,
                    help='Override the tool offset: x,y,z[,a,b,c] mm/deg, '
                         'flange -> fingertip. For trying a number before '
                         'it is properly measured.')
    ap.add_argument('--min-seg-s', type=float, default=0.5,
                    help='Drop feasible spans shorter than this (s).')
    ap.add_argument('--bridge-s', type=float, default=0.05,
                    help='Swallow infeasible spans shorter than this (s) '
                         'into the surrounding segment.')
    ap.add_argument('--max-lag-mm', type=float, default=5.0,
                    help='Tip lag past which a clamped frame counts as '
                         'over-budget rather than merely slow.')
    ap.add_argument('--max-rot-lag-deg', type=float, default=3.0)
    ap.add_argument('--no-keepout', dest='keepout', action='store_false',
                    help='Ignore config.UMI_REPLAY_KEEPOUT (which is a '
                         'provisional box, not a measured body envelope).')
    ap.add_argument('--no-recheck', dest='recheck', action='store_false',
                    help='Skip re-running each kept span from the start pose '
                         'on its own.')
    ap.add_argument('--no-resync', action='store_true',
                    help='Stop at the first fault instead of re-anchoring — '
                         'answers "how much of this survives in one take".')
    ap.add_argument('--sweep', nargs='*', type=float, default=None,
                    metavar='SPEED',
                    help='Also scan at these replay speeds and print a table '
                         '(default 1.0 0.7 0.5 0.3).')
    ap.add_argument('--out', default=None,
                    help='Write the mask/report JSON here.')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if args.speed <= 0:
        print('❌ --speed 必须为正'); return 2
    if args.sweep is not None and not args.sweep:
        args.sweep = [1.0, 0.7, 0.5, 0.3]
    config.SCALE = args.scale
    if args.vel_ratio is not None:
        config.apply_vel_ratio(args.vel_ratio)

    raw, info = load_episode(args.dataset, args.episode)
    hands = [h for h in HANDS if raw[0].pose(h) is not None]
    arms = list(args.arms.upper()) if args.arms \
        else [config.ARM_OF_HAND[h] for h in hands]
    for a in arms:
        if a not in HAND_OF_ARM:
            print('❌ 未知的臂 %r' % a); return 2
        if HAND_OF_ARM[a] not in hands:
            print('❌ 臂 %s 需要 %s 手的数据，episode 里没有'
                  % (a, HAND_OF_ARM[a])); return 2

    frames = resample(raw, config.CONTROL_HZ / args.speed)
    print('载入 %d 帧 @ %sHz ← %s (episode %d, %s)  有效手: %s'
          % (len(raw), info.get('fps'), args.dataset, args.episode,
             info.get('robot_type'), hands))
    print('插值到 %d 个控制周期 @ %.0fHz  =  %.2f 秒回放（%.2fx 速度）'
          % (len(frames), config.CONTROL_HZ, len(frames) / config.CONTROL_HZ,
             args.speed))
    config.summary()

    tools, starts, notes, iks = {}, {}, {}, {}
    args.tool_note_unset = False
    for a in arms:
        tools[a], notes[a] = resolve_tool(args, a)
        starts[a], src = resolve_start(args, a)
        iks[a] = emb.make_ik(a, config)
        if tools[a] is None:
            args.tool_note_unset = True
        print('臂%s 起始构型 %s ← %s' % (a, [round(v, 1) for v in starts[a]],
                                        src))
        print('     指尖偏移 %s ← %s'
              % ('[%.1f %.1f %.1f] mm，离法兰 %.0f mm'
                 % (tuple(tools[a].xyzabc[:3]) + (tools[a].reach_mm,))
                 if tools[a] else '⚠️ 未设置，按法兰判定', notes[a]))
    if args.tool_note_unset:
        print('\n⚠️  指尖未对齐：录制的位姿是 leader 夹爪的**两指中点**，而这里的 '
              'FK 给的是**法兰**。\n'
              '    纯平移时两者等价（相对映射把常数偏移抵消掉了），一旦有腕部'
              '旋转就不等价 ——\n'
              '    这段 demo 峰值 106°/s，判的就不是将来真正会跑的那条轨迹。\n'
              '    先 scripts/tip_calib.py 标定，或用 --tip 临时给个值。')

    t0 = time.time()
    scans = []
    for a in arms:
        sc = run_arm(frames, raw, a, config, args, tools[a], starts[a], iks[a])
        if args.recheck:
            recheck(sc, frames, a, config, args, tools[a], starts[a], iks[a])
            # A demoted span can leave two cut spans adjacent; re-merge so the
            # report keeps segment()'s invariant.
            sc.segments = emb.stamp_times(emb.merge_cuts(sc.segments),
                                          frames, raw)
        scans.append(sc)
        report(sc, config, args)

    keep = emb.intersect([spans(s) for s in scans], len(frames))
    if len(scans) > 1:
        kept = sum(b - a for a, b in keep)
        print('\n=== 双臂交集 ===')
        print('两条臂都能跟的帧：%d/%d = %.1f%%，共 %d 段'
              % (kept, len(frames), 100.0 * kept / max(len(frames), 1),
                 len(keep)))
    # Recorded-frame ranges are what a dataset consumer can actually use.
    keep_segs = emb.stamp_times([emb.Segment(a, b, True) for a, b in keep],
                                frames, raw)
    if keep_segs:
        print('\n可用片段（录制帧下标，可直接喂给 --frames）：')
        for s in keep_segs:
            print('  --frames %d:%d      # %.2f..%.2fs, %d 个录制帧'
                  % (s.rec_start, s.rec_end, s.t_start, s.t_end,
                     s.rec_end - s.rec_start))

    advise(scans, config, args)
    if args.sweep:
        sweep(frames, raw, arms, config, args, tools, starts, iks)
    print('\n扫描用时 %.1f 秒' % (time.time() - t0))

    if args.out:
        doc = {
            'dataset': os.path.abspath(args.dataset),
            'episode': args.episode,
            'fps': info.get('fps'),
            'control_hz': config.CONTROL_HZ,
            'generated_by': 'scripts/umi_filter.py',
            'settings': {
                'scale': config.SCALE, 'speed': args.speed,
                'vel_ratio': config.VEL_RATIO,
                'max_joint_rate_deg_s': config.MAX_JOINT_RATE_DEG_S,
                'min_seg_s': args.min_seg_s, 'bridge_s': args.bridge_s,
                'max_lag_mm': args.max_lag_mm,
                'max_rot_lag_deg': args.max_rot_lag_deg,
                'keepout': bool(args.keepout), 'recheck': bool(args.recheck),
                'resync': not args.no_resync,
            },
            'arms': {},
            'keep_recorded': [[s.rec_start, s.rec_end] for s in keep_segs],
        }
        for sc in scans:
            doc['arms'][sc.arm] = {
                'hand': sc.hand,
                'start_joints': sc.q0,
                'tool_offset': sc.tool.xyzabc if sc.tool else None,
                'tool_source': notes[sc.arm],
                'frames': sc.n,
                'feasible': sc.n_feasible,
                'counts': sc.counts(),
                'breaks': sc.breaks,
                'worst_margin_deg': sc.worst_margin,
                'max_rate_ratio': sc.max_rate_ratio,
                'max_lag_mm': sc.max_lag_mm,
                'segments': [{
                    'keep': s.ok, 'start': s.start, 'end': s.end,
                    'rec_start': s.rec_start, 'rec_end': s.rec_end,
                    't_start': round(s.t_start, 4), 't_end': round(s.t_end, 4),
                    'reasons': s.reasons, 'bridged': s.bridged,
                    'worst_margin_deg': s.worst_margin,
                    'max_rate_ratio': s.max_rate_ratio,
                    'max_lag_mm': s.max_lag_mm,
                    'tip_lo': s.lo, 'tip_hi': s.hi,
                    'recheck': s.recheck,
                } for s in sc.segments],
            }
        with open(args.out, 'w') as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        print('已写入 %s' % args.out)

    # Exit non-zero when nothing survives -- this runs in scripted curation.
    return 0 if keep_segs else 1


if __name__ == '__main__':
    sys.exit(main())
