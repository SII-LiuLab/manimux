"""Move each arm back to a known configuration at the end of a run.

See docs/core.md for design rationale and empirical data.
"""
import config
from drivers.arm_driver import STATE_ERROR, STATE_POSITION, move_to_joints


def home_arms(conn, channels, targets_by_arm=None, label='归位'):
    """Move each arm to a parking configuration before servos are disabled.

    :param targets_by_arm: {arm: [7 angles]}; defaults to config.HOME_JOINTS.
        A pipeline whose safe parking pose is not HOME passes its own --
        scripts/umi_replay.py parks at config.UMI_START_JOINTS, because HOME
        does not have the clearance that pipeline's trajectory needs and
        parking there would leave the next run unable to start from where it
        stopped. See umi与天机replay.md.
    """
    table = config.HOME_JOINTS if targets_by_arm is None else targets_by_arm
    if conn.dry_run:
        print('\n[dry-run] 跳过%s' % label)
        return
    drivers = {a: c.driver for a, c in channels.items()}
    targets = {}
    for arm in channels:
        if arm not in table:
            print('⚠️  臂%s 没有配%s目标构型，跳过' % (arm, label))
            continue
        st = drivers[arm].state()
        # Don't command motion on a faulted or non-position-mode arm.
        if st['err'] or st['cur'] == STATE_ERROR:
            print('⚠️  臂%s 处于错误状态（状态 %s 错误码 %s），跳过%s'
                  % (arm, st['cur'], st['err'], label))
            continue
        # Checked against the hardcoded position state, not config.ARM_STATE
        # -- config.ARM_STATE may be torque (impedance runs), but homing's
        # rate-limited interpolation (move_to_joints) has only ever been
        # validated in position mode; torque-mode homing is untested, so
        # this must always require position mode regardless of what mode
        # the run used. See docs/core.md.
        if st['cur'] != STATE_POSITION:
            print('⚠️  臂%s 已不在位置模式（状态 %s），跳过%s'
                  % (arm, st['cur'], label))
            continue
        targets[arm] = list(table[arm])
    if not targets:
        return

    print('\n=== %s ===' % label)
    ok, why, q_end = move_to_joints(conn, drivers, targets,
                                    speed_deg_s=config.HOME_SPEED_DEG_S,
                                    hz=config.CONTROL_HZ,
                                    max_err_deg=config.MAX_TRACKING_ERR_DEG)
    for arm in targets:
        q = drivers[arm].joints()
        print('臂%s 现构型 %s' % (arm, [round(v, 2) for v in q]))
    if not ok:
        print('⚠️  %s未完成（%s）—— 下次开跑前先跑 scripts/goto_joints.py'
              % (label, why))
        return
    print('✅ 已到位（%s，失能后会保持在此姿态）' % label)
