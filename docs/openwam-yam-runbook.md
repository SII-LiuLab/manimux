# OpenWAM on YAM through ManiMux

## Status

ManiMux integrates OpenWAM as an external WebSocket policy. The model and its
CUDA dependencies stay in `/home/ubuntu/OpenWAM`; ManiMux owns robot state,
camera acquisition, action timing, IK, smoothing, and recording.

The downloaded `OpenWAM-Alpha-Pretrain-Foundation-Model` is **fine-tune only**.
It has no YAM normalization artifact and must not be passed to `deploy.sh` as a
YAM controller. A YAM fine-tune must preserve the contract below and produce a
self-contained checkpoint with `normalization_stats.npy`.

## YAM representation contract

The native reader input and server output are 20-D absolute EEF values:

```text
[left_xyz_m(3), left_rot6d(6), left_gripper(1),
 right_xyz_m(3), right_rot6d(6), right_gripper(1)]
```

- `xyz` is in metres in each arm's robot-base frame.
- `rot6d` is `[R[:, 0], R[:, 1]]` and is not min-max transformed.
- Native gripper is `0=closed, 1=open`. OpenWAM training min-max maps that to
  `[-1, 1]`; the server must inverse-normalize it back to `[0, 1]`.
- The fine-tuned checkpoint config must advertise `action_mode: eef` and
  `gripper_convention: zero_closed_one_open`; ManiMux rejects other handshakes.
- Use YAM training-split statistics. Do not reuse ARX-X5 statistics.
- Map native left `0:10` to alpha slots `0:10` and native right `10:20` to
  alpha slots `34:44`.
- Use `num_frames=33`, `video_stride=4`, and a 32-step action horizon.

The existing LeRobot converter can preserve absolute observation and command
EEF poses with `--include-ee-pose`, but OpenWAM still needs a dedicated reader
that converts the stored rotation matrices to rot6d, builds EEF20, fits the
YAM min-max statistics, and scatters it into the alpha 80-D space.

## Serving a YAM-finetuned checkpoint

Install OpenWAM in a separate environment following its upstream README. Start
the ManiMux full-chunk entry point with that environment:

```bash
cd /home/ubuntu/manimux
/home/ubuntu/OpenWAM/.venv/bin/python scripts/servers/openwam_yam_chunk_server.py \
  --ckpt-dir /path/to/yam_finetuned_checkpoint \
  --device cuda:0 \
  --port 8848 \
  --inference-mode sync \
  --compile-enabled false
```

`--compile-enabled false` is useful for the first compatibility pass. Re-enable compile
only after eager inference is validated. The server must be dedicated to one
ManiMux session because reset state and request numbering are process-global.

After a non-hardware server smoke check, the ManiMux configuration is:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/openwam/yam/infra/manimux.yaml
```

The server returns the official engine's full 32-step chunk in one response.
The `openwam_yam` adapter then converts each absolute EEF target to a
joint-position waypoint with YAM IK. `response_mode: stream` remains available
for protocol experiments against the unmodified official server, but it should
not be used for real-time YAM execution. This is ordinary chunk inference;
OpenWAM's asynchronous executor is not ManiMux RTC.

The example keeps `execution.max_plan_age_s: 2.0`; results older than that are
rejected. Adjust this value from measured eager/compiled inference latency for
the target task.
