#!/usr/bin/env python3
"""Single source of truth for every tunable in the teleop pipeline.

See docs/config.md for calibration history, empirical measurements, and
tuning rationale behind these values.
"""

# ---- Hardware ----
ROBOT_IP = '192.168.1.190'

ARMS = ['A', 'B']  # 'A' = left, 'B' = right

ARM_OF_HAND = {'left': 'A', 'right': 'B'}

# Parking pose after each run. See docs/config.md#hardware.
HOME_JOINTS = {
    'A': [90.0, -90.0, -90.0, -90.0, 0.0, 0.0, 0.0],
    'B': [-90.0, -90.0, 90.0, -90.0, 0.0, 0.0, 0.0],
}

# Intermediate poses scripts/home_now.py passes through, in order, before
# the final HOME_JOINTS move -- one move_to_joints() call per stage, so
# each leg is checked for tracking error independently. Arms with no entry
# here (or not present) skip straight to HOME_JOINTS.
# 'B' is hardware-verified. 'A' is a candidate, NOT YET HARDWARE-VERIFIED --
# verify with scripts/goto_joints.py --arm A --to=... --speed 3 before
# trusting it in home_now.py's normal path.
HOME_WAYPOINTS = {
    'A': [[75.0, -10.0, -20.0, -80.0, 0.0, 0.0, 0.0]],
    'B': [[-75.0, -10.0, 20.0, -80.0, 0.0, 0.0, 0.0]],
}

HOME_SPEED_DEG_S = 6.0

# This is a 4.0 machine, so this must be the m6_40 file, NOT m6_31. The two
# files' DH and PNVA tables are bit-identical (so FK/IK is unaffected either
# way, which is why the m6_31 mistake survived every kinematics test we have),
# but m6_31 is an older *file layout*: it stores each joint row as
# inertia,mass,com while m6_40 stores mass,com,inertia, and the shipped
# LOADMvCfg parser only knows the m6_40 layout. Loading m6_31 therefore
# "succeeds" and silently returns garbage dynamics -- Mass comes back as
# [0.105, 0.027, 0, 0, 0.001, 0, 0.002] kg (those are inertia terms; four
# joints weigh zero) with the real 4.04 kg sitting in the inertia slot. Only
# the dynamics paths read those, but nothing warns you. See also BD67_REAL in
# algos/ik_solver.py: the hardware's live BD67 table equals m6_40's exactly and
# matches m6_31's not at all -- that is what identifies the machine as 4.0.
KINE_CFG = ('/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK/'
            'CommonConfig/ccs_m6_40.MvKDCfg')


# ---- Coordinate calibration ----
# See docs/config.md#coordinate-calibration for derivation and caveats.
CALIBRATED = True

# XR raw pose -> right-handed frame. None = no flip.
XR_HANDEDNESS_FLIP_AXIS = None

XR_POS_TO_MM = 1000.0  # XR position unit is meters

XR_QUAT_ORDER = 'xyzw'

# Right-handed XR -> robot base frame, per arm. Keys are XR basis vectors,
# not semantic names — see retarget.build_axis_matrix().
AXIS_MAP = {
    'A': {'xr_x': '-Z', 'xr_y': '-Y', 'xr_z': '-X'},   # measured
    'B': {'xr_x': '+Z', 'xr_y': '+Y', 'xr_z': '-X'},   # derived, unverified
}

# Same thing for the UMI rig (drivers/umi_source.py). Its poses are NOT in the
# raw PICO frame: xense-taccap-lerobot's Pico4TrackerReader already applies
# PICO_TO_WORLD_R, remapping [x,y,z] -> [-z,-x,y] into a gravity-aligned world
# frame (X forward, Y left, Z up). So this is not a second, independent
# calibration -- it is AXIS_MAP composed with that known remap:
#
#     v_base = AXIS_MAP · v_pico          (the rows above)
#     v_world = G · v_pico                (PICO_TO_WORLD_R)
#     =>  v_base = (AXIS_MAP · Gᵀ) · v_world      <- the rows below
#
# NOTE (delta_frame): no longer read by the replay path. scripts/umi_replay.py
# and replay.py --umi now take the relative motion in the gripper's own EE
# frame (Retargeter delta_frame='ee'), matching the policy pipeline and
# deployment, and a body-frame delta has no world->base axis map to apply.
# Kept because it is still the documented derivation of the PICO-world->base
# relationship and the one thing to re-derive if that path ever comes back.
#
# Verified numerically in drivers/umi_source.py's docs (docs/drivers.md).
# It inherits AXIS_MAP's status exactly: 'A' traces back to a measured
# mapping, 'B' to a derived and still unverified one -- so a wrong-looking
# arm B result should suspect AXIS_MAP['B'] first, not this table.
AXIS_MAP_UMI = {
    'A': {'xr_x': '+X', 'xr_y': '+Z', 'xr_z': '-Y'},
    'B': {'xr_x': '+X', 'xr_y': '-Z', 'xr_z': '+Y'},
}


# ---- Retargeting ----
SCALE = 0.5             # hand-motion -> robot-motion ratio
FOLLOW_ROTATION = True  # False = translation only, orientation locked

CLUTCH_BUTTON = 'grip'
CLUTCH_THRESHOLD = 0.5

GRIPPER_AXIS = 'trigger'

# Manual arm-angle (nullspace) control: stick X -> arm-angle rate (deg/s).
# Additive offset on top of the automatic term, not exclusive with it.
ARM_ANGLE_AXIS = 'axisX'
ARM_ANGLE_RATE = 20.0
ARM_ANGLE_LIMIT = 45.0  # travel from latched pose, deg

# ---- Nullspace joint-limit avoidance (see nullspace.py) ----
NULLSPACE_ENABLED = True
NULLSPACE_ACTIVATION_DEG = 25.0     # deadband: engage only this close to a limit
NULLSPACE_RATE_DEG_S = 20.0         # arm-angle slew rate
NULLSPACE_PROBE_DEG = 0.5           # gradient probe step

LOWPASS_HZ = 8.0  # controller-pose low-pass cutoff

# CART_MAX_SPEED_MM_S / CART_MAX_ROT_DEG_S are derived below in the
# speed-budget section, kept consistent with the joint-side limits.


# ---- Control loop ----
CONTROL_HZ = 250.0
XR_STALE_S = 0.25  # freeze if no new XR frame for this long

ARM_STATE = 1  # 1 = position follow (default), 3 = torque (--control-mode
               # impedance flips this; see run_teleop.py). Impedance
               # type/K/D moved to algos.solver_config.ImpedanceConfig /
               # configs/solver/impedance.yaml -- not hardware-validated
               # production values yet, so they don't belong in config.py
               # (see solver_config.py's own docstring on this).


# ---- Speed budget ----
# Every speed limit in the chain is derived from VEL_RATIO. See
# docs/config.md#speed-budget for why these must stay centralized.

JOINT_VMAX_DEG_S = 180.0  # from ccs_m6_40.MvKDCfg PNVA table

# Controller-side follow velocity percent. Edit this value directly; the
# derived quantities below recompute on import. apply_vel_ratio() is for
# runtime changes only (the --vel-ratio CLI flag). Pick a value that
# round-trips through the controller's float32 percent readback -- some
# (e.g. 65, 69, 70) read back one lower and fail _check_vel_ratio.
VEL_RATIO = 32

# Acceleration percent — deliberately decoupled from VEL_RATIO and set
# much higher; governs servo ramp-up lag, not top speed.
ACC_RATIO = 100

# Command-stream margin below controller capability. Offline sweeps can
# show clamp rate but not real tracking-error risk (replay.py/bench assume
# perfect servo tracking -- see docs/core.md's dry-run caveat), so raise
# this only in small steps verified live, same as VEL_RATIO gear changes:
# watch peak tracking error / protection-lockout count before going further.
CMD_RATE_MARGIN = 0.90

# ---- Derived — change VEL_RATIO via apply_vel_ratio(), not directly ----
MAX_JOINT_RATE_DEG_S = JOINT_VMAX_DEG_S * VEL_RATIO / 100.0 * CMD_RATE_MARGIN
MAX_JOINT_STEP_DEG = MAX_JOINT_RATE_DEG_S / CONTROL_HZ
MAX_STEP_DT_S = 4.0 / CONTROL_HZ  # cap banked dt to 4 periods

NOMINAL_DEG_PER_MM = 0.2  # mid-range mm->deg conversion, see docs/config.md
CART_MAX_SPEED_MM_S = MAX_JOINT_RATE_DEG_S / NOMINAL_DEG_PER_MM
CART_MAX_ROT_DEG_S = MAX_JOINT_RATE_DEG_S * 1.2


def apply_vel_ratio(vel, acc=None):
    """Change speed gear. Recomputes all four derived speed-budget values."""
    global VEL_RATIO, ACC_RATIO, MAX_JOINT_RATE_DEG_S, MAX_JOINT_STEP_DEG
    global CART_MAX_SPEED_MM_S, CART_MAX_ROT_DEG_S
    vel = int(vel)
    if not 1 <= vel <= 100:
        raise ValueError('VEL_RATIO must be in 1..100, got %r' % vel)
    VEL_RATIO = vel
    if acc is not None:
        ACC_RATIO = int(acc)
    MAX_JOINT_RATE_DEG_S = JOINT_VMAX_DEG_S * VEL_RATIO / 100.0 * CMD_RATE_MARGIN
    MAX_JOINT_STEP_DEG = MAX_JOINT_RATE_DEG_S / CONTROL_HZ
    CART_MAX_SPEED_MM_S = MAX_JOINT_RATE_DEG_S / NOMINAL_DEG_PER_MM
    CART_MAX_ROT_DEG_S = MAX_JOINT_RATE_DEG_S * 1.2


# ---- Safety thresholds ----
# Per-frame joint-step ceiling is in the speed-budget section above.

# Inside robot.ini's VelLmtRange=8, so the controller's own rate limiting can
# engage in the last 3 deg. See docs/config.md#safety-thresholds.
LIMIT_MARGIN_DEG = 5.0

# FK back-substitution tolerance for an IK solution — last-line defense
# against a false-success solution. See docs/config.md#safety-thresholds.
IK_POS_TOL_MM = 0.01
IK_ROT_TOL_DEG = 0.2

# IK-layer branch-jump detector (not a velocity limit — that's the gate's job).
IK_MAX_STEP_DEG = 1.8

MAX_TRACKING_ERR_DEG = 5.0  # command-vs-measured deviation ceiling
MAX_TRACKING_ERR_S = 0.5    # how long it must stay over before it's a fault

MAX_REJECT_S = 0.5  # consecutive-IK-rejection time before disengage
MAX_CONSEC_REJECT = int(MAX_REJECT_S * CONTROL_HZ)

# Arm B joint 6 limit unconfirmed (tools/move_one_joint.py says +-58,
# config file says +-60); using the smaller value pending verification.
JOINT_LIMIT_OVERRIDE = {
    'B': {5: (-58.0, 58.0)},
}


# ---- UMI replay start pose ----
# The UMI reference stack splits this in two: bimanual_umi_env.py holds a fixed
# `j_init` (UR5e `[0,-90,-90,-90,90,0]`) that only has to start
# well-conditioned, and eval_real.py then has the operator jog to a
# task-appropriate start with a SpaceMouse before handing over. We have no
# SpaceMouse, so this constant does both jobs.
#
# Deliberately NOT derived from HOME. HOME is a parking pose -- folded in, TCP
# only 174.5mm off the centerline -- and this episode's inward sweep is 152.8mm
# (arm A) / 95.6mm (arm B) in base -Z, so starting there ends 22mm from the
# body. That is the 2026-08-25 near-collision. This is a "ready to work" pose
# instead: reaching forward and spread outward, 219mm from HOME's TCP.
#
# Found by grid search over the reachable space at 10-degree resolution
# (330750 arm configurations x neat wrist combinations), ranked by
# manipulability subject to a joint-limit margin >= 25 deg and a TCP box. Every
# joint is a multiple of 10 by construction -- these are read off a terminal
# and typed back in, and 88.23 is a transcription error waiting to happen.
#
# B is A mirrored (negate J1/J3/J5/J7, keep J2/J4/J6) -- the rule is verified
# against both HOME_JOINTS and HOME_WAYPOINTS, and reproduces a Y-negated TCP
# exactly.
#
# Re-derive when the episode changes: the clearance requirement belongs to the
# demo, not to the robot. scripts/umi_replay.py's pre-flight prints the number.
#
# 2026-08-25: replaced by the pose the arms were jogged to by hand and read
# back off scripts/get_current_pos.py. This is the SpaceMouse half of the UMI
# split above (operator picks a task-appropriate start), so it is a hand-set
# value, not a grid-search result -- the metrics below were measured on it
# afterwards, they did not select it.
#
# J6 was 3 / -3 as jogged, which breaks the mirror rule above and left the two
# TCPs 7.5mm apart in Y; zeroed on both arms, which restores an exact mirror
# (identical w / sigma_min / margin, TCP Y +-198.77) at no measurable cost.
#
#   pose                     w      sigma_min  limit margin  TCP (base frame)
#   A [50,-40,-30,-100,-65,0,40]   266.7  0.771  45 deg (J4)  [+464.6 +198.8 +177.4]
#   B mirrored                     266.7  0.771  45 deg (J4)  [+464.6 -198.8 +177.4]
#   previous grid-search pose      312.8  0.782  40/38 deg    [+486.7 +-293.9 +384.4]
#
# What it buys: J4 gains enough margin that replay_test_2 solves 100% of frames
# on BOTH arms even at scale 1.0 -- the grid-search pose ran arm A into the J4
# limit for 296 frames there (and J4 is the one joint the nullspace controller
# has zero leverage over, see docs/config.md).
#
# What it costs -- MEASURED on replay_test_2, offline, and it is not good:
#   * TCP starts 177mm off the centerline, down from 384mm, i.e. back to
#     roughly HOME's 174.5mm. That is the geometry the 2026-08-25
#     near-collision came out of.
#   * closest approach, base-frame Z:  scale 0.5 -> A 91.0mm, B 74.9mm
#                                      scale 1.0 -> A 53.6mm, B -5.8mm
#     against the 100mm keep-out floor the grid-search pose cleared by 232/289mm.
#     Every one of those is under the floor; B at scale 1.0 crosses the plane.
#     The 2026-08-25 incident itself was 22mm.
# Lower manipulability too (267 vs 313), which matters much less than the above.
#
# So: fine for episodes whose inward sweep is small, NOT for replay_test_2.
# UMI_REPLAY_KEEPOUT below is still None, so the pre-flight will print these
# numbers and let the run proceed anyway. Fill it before trusting this pose.
UMI_START_JOINTS = {
    'A': [50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0],
    'B': [-50.0, -40.0, 30.0, -100.0, 65.0, 0.0, -40.0],
}

UMI_START_SPEED_DEG_S = 6.0     # same gear as HOME_SPEED_DEG_S


# ---- UMI replay workspace guard ----
# Per-arm keep-out box in the arm's BASE frame, mm: {'X': (lo, hi), ...}.
# scripts/umi_replay.py simulates the whole episode before sending anything
# and refuses to start if the TCP envelope leaves this box.
#
# Why this exists: nothing else in this stack models self-collision. The
# safety gate checks joint limits, joint rate and tracking error -- all of
# which passed on 2026-08-25 while arm A drove its TCP from Z=+174mm to
# Z=+22mm, i.e. onto the robot's own centerline. Relative mapping reproduces
# the demonstrator's displacement, not their clearance.
#
# Arm A base frame is X forward / Y down / Z left (derived from AXIS_MAP);
# for arm A, SMALL Z = close to the body. Arm B is mirrored (Z right), so its
# body side is LARGE Z -- do not copy A's numbers to B.
#
# None = unguarded (only the printed envelope protects you). Fill in measured
# numbers for this cell before trusting an unattended run.
# PROVISIONAL, NOT MEASURED. The 100mm Z floor comes from one observation --
# the operator hit e-stop on 2026-08-25 with arm A's TCP heading for Z=+22mm --
# not from a measured body envelope. It is set because an unguarded default
# offers nothing, but replace it with real numbers for this cell before
# trusting it. Both arms sweep toward SMALL Z (their frames are mirrored, so
# 'inward' is -Z for both); X/Y are left unguarded for the same honesty reason.
UMI_REPLAY_KEEPOUT = {
    'A': {'Z': (100.0, 900.0)},
    'B': {'Z': (100.0, 900.0)},
}


# ---- Gripper: UMI / TacCap follower (drivers/umi_gripper.py) ----
# This is what is physically mounted on both arms now. It does NOT go through
# Marvin's CAN pass-through -- it is a separate USB serial link per side
# (CH343 -> MCU -> FDCAN -> motor), so none of the GRIPPER_* values below
# apply to it and none of these apply to the OmniGripper.
#
# Convention here is the SDK's: 0 = CLOSED, 1 = OPEN. That is the INVERSE of
# GRIPPER_* below, and it is deliberate -- recorded UMI episodes use the SDK
# convention too (drivers/umi_source.py), so the recorded jaw value feeds
# set_target() with no conversion at all. Anything that still speaks the old
# convention has to flip, not this.

UMI_GRIPPER_ENABLED = False     # --gripper flips this on the UMI replay path

# Pin the firmware SN per arm once the grippers are labelled. None =
# auto-discover by the SDK's side rule. Pinning is strongly preferred: a
# changed USB enumeration order can otherwise silently swap left/right, and
# a gripper driven as the wrong side is an expensive mistake to find out
# about on hardware.
UMI_GRIPPER_SN = {'A': None, 'B': None}

UMI_GRIPPER_HZ = 100        # ControlLoop submit rate; STREAM_LOCKED phases it
                            # to the motor status stream, see docs/drivers.md
UMI_GRIPPER_KP = 8.0        # N*m/rad, SDK default
UMI_GRIPPER_KD = 0.3        # N*m*s/rad

# Max lead of commanded over measured jaw position, in NORMALIZED units.
# Grip force ~= UMI_GRIPPER_KP * (this * stroke_rad); the driver prints the
# implied N*m at connect time. Same idea as GRIPPER_MAX_OVERSHOOT_RAD below,
# completely different numbers. 0.036 commands 8.0 * 0.036 * 1.20 rad =
# 0.346 N*m, level with the firmware auto-cal's own 0.350 N*m stall torque
# (GripperAutoCalConfig). The SDK example's 0.045 assumes kp 5.0.
UMI_GRIPPER_GRIP_MARGIN = 0.036

# Abort-and-disable torque. 1.0 by operator decision on 2026-09-10 -- not
# spec-backed (no motor rating is published); it sits well above the 0.346
# N*m commanded grip, so it only catches a hard jam. Grip force is
# GRIP_MARGIN's job. The SDK example's 0.30 is below the firmware's own
# 0.350 stall torque and fires on normal grasps.
UMI_GRIPPER_MAX_TAU = 1.0

# Staleness ceiling on ControlLoop.observation(). The motor's actual_*
# telemetry refreshes at ~50-100Hz, so anything past ~200ms means the link
# is gone, not merely slow.
UMI_GRIPPER_STALE_MS = 200.0


# ---- Gripper: OmniGripper / DM4310 (drivers/gripper.py) ----
# DEPRECATED for the current hardware -- kept for the OmniGripper, which is
# no longer mounted on either arm (see data/tool_calib/README.md). Convention
# is 0 = OPEN, 1 = CLOSED, the inverse of the UMI block above.
GRIPPER_ENABLED = False
GRIPPER_HZ = 50.0  # own thread, not the joint control loop

# Measured travel (scripts/gripper_range.py). See docs/config.md#gripper.
GRIPPER_OPEN_RAD = -0.02
GRIPPER_CLOSE_RAD = 1.15
# kp=10 -> full-stroke close ~0.585s; kp=20 -> ~0.395s (kp=30/40 no faster,
# see scripts/gripper_speed_test.py). Worst-case stall force = kp *
# GRIPPER_MAX_OVERSHOOT_RAD ~= 4.0 N*m at kp=20, still well under the
# vendor DM4310 TAU_MAX=10.0.
GRIPPER_KP = 10
GRIPPER_KD = 0.5
# Fault ceiling. Vendor DM4310 TAU_MAX=10.0 N*m (protocol/stall limit, not
# a "safe to sustain" value) -- kept 2N*m below it so a genuine jam still
# trips a software fault before the motor sits at its hard limit. Normal
# worst-case commanded torque tops out at ~6.0 (see GRIPPER_MAX_OVERSHOOT_RAD
# below), so this still has margin above expected operation too.
GRIPPER_MAX_TAU = 8.0
GRIPPER_MAX_ERR_RAD = 0.5

# Force-limited position control: max lead of commanded over measured
# position (rad). Grip torque ~= GRIPPER_KP * GRIPPER_MAX_OVERSHOOT_RAD;
# at this value, worst-case stall torque is 20*0.3=6.0N*m, 2N*m below
# GRIPPER_MAX_TAU's fault line (8.0) and 4N*m below vendor TAU_MAX (10.0).
GRIPPER_MAX_OVERSHOOT_RAD = 0.3
# >= travel/GRIPPER_HZ (1.17/50 ~= 58.5) so the slew ramp stops being the
# bottleneck -- GRIPPER_MAX_OVERSHOOT_RAD is what actually governs close
# speed/force from here, not this. See docs/config.md#gripper.
GRIPPER_SLEW_RAD_S = 20


def summary():
    """Print key config for a visual pre-flight check."""
    ctrl = JOINT_VMAX_DEG_S * VEL_RATIO / 100.0
    print('机器人 %s  手臂 %s  模式 %s' % (ROBOT_IP, ARMS, ARM_STATE))
    print('标定 %s  scale %.2f  跟随姿态 %s'
          % ('已完成' if CALIBRATED else '⚠️ 未标定', SCALE, FOLLOW_ROTATION))
    print('控制 %.0fHz  vel/acc %d%%/%d%%' % (CONTROL_HZ, VEL_RATIO, ACC_RATIO))
    print('速度预算: 控制器 %.0f°/s  →  指令流 %.0f°/s (%.0f%%)'
          '  =  %.3f°/帧 @%.0fHz' % (ctrl, MAX_JOINT_RATE_DEG_S,
                                     CMD_RATE_MARGIN * 100,
                                     MAX_JOINT_STEP_DEG, CONTROL_HZ))
    print('          笛卡尔 %.0f mm/s / %.0f °/s'
          % (CART_MAX_SPEED_MM_S, CART_MAX_ROT_DEG_S))
    if MAX_JOINT_RATE_DEG_S >= ctrl:
        print('❌ 指令流限幅 %.0f°/s ≥ 控制器能力 %.0f°/s —— 瓶颈又倒置了，'
              '跟踪误差会累积到 tracking_error 闭锁。检查 CMD_RATE_MARGIN。'
              % (MAX_JOINT_RATE_DEG_S, ctrl))
    print('          末端参考速度: %.0f mm/s @0.13°/mm(好方向)  '
          '%.0f mm/s @0.45°/mm(近边界)'
          % (MAX_JOINT_RATE_DEG_S / 0.133, MAX_JOINT_RATE_DEG_S / 0.454))
