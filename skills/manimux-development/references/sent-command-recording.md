# Submitted and sent YAM commands

Training continues to use the teleoperation `action-*` fields. These are targets
submitted to the follower SDK. The diagnostic fields below never replace them.

The i2rt patch at
`manimux/embodiments/arm/yam/patches/i2rt-sent-command.patch` captures the latest
complete MIT command cycle at the CAN driver. It includes SDK joint limits,
gripper force limiting, motor offsets/directions and MIT position quantization.
It has been applied to this station's collection and runtime SDK copies. After
reinstalling an SDK, apply the patch from the directory containing `i2rt/` using
`git apply --check` first. Existing Python processes need a normal restart to load
the patch; this change does not restart them or command hardware.

## Saved fields

Collection saves `sent-<arm>-<field>.npy`, replacing underscores in field names
with hyphens. Runtime saves `data.zarr/ticks/sent_command/<group>/<field>`.

| Field | Meaning |
| --- | --- |
| `position` | Decoded wire target in joint radians and normalized gripper opening |
| `motor_position` | Decoded MIT position in motor radians |
| `send_time_ns` | Unix time immediately before the last successful host `bus.send` for each motor |
| `send_monotonic_ns` | Monotonic time immediately before the same call |
| `send_end_monotonic_ns` | Monotonic time after that call returns |
| `send_count` | Successful host sends, including retries, in that motor transaction |
| `sequence` | SDK-local complete-cycle counter |
| `valid` | Whether a completed send snapshot was available |

Position and timestamp arrays have one column per motor, including the gripper.
Missing snapshots use `valid=false`, `NaN` positions and `-1` integer fields.
An unpatched SDK remains usable but cannot supply this evidence.

These are sampled snapshots, not a lossless CAN log. Several rows can refer to the
same cycle, and intermediate cycles can be skipped. A snapshot can precede the
current row's submitted action; use its timestamps and sequence rather than
assuming a causal row match. An incomplete/failed cycle does not replace the last
complete snapshot. Host `bus.send` time is not a hardware receive timestamp.

FrameLab displays submitted action, CAN-sent position and measured state as
separate curves. Legacy episodes have no CAN-sent curve; no synthetic substitute
is inserted. CAN timestamps and SDK target-cache timestamps are reported separately.
