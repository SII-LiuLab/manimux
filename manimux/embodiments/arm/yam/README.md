# Integrated YAM arm and gripper component

`YamArm` / `YamController` in `arm.py` call the installed i2rt directly, without another
`YAMRobot` wrapper or a copied SDK. `kinematics.py` provides the complete TCP interface,
existing MuJoCo FK / i2rt-Mink IK and visual coordinate mapping; model files live in `assets/`.

## Environment

Run from the repository root. Create a new environment as follows, or install dependencies using an existing environment's interpreter:

```bash
uv venv --python 3.12 envs/yam/.venv
uv pip install --python envs/yam/.venv/bin/python -e '.[replay,realsense,xpolicylab]'
uv pip install --python envs/yam/.venv/bin/python \
  'git+https://github.com/i2rt-robotics/i2rt.git@5d47b358bafb30c65e397f2ece506550a0db4594'
```

The SDK baseline is i2rt 1.1.2 at the revision above. The SDK manages CAN, motor loops, gravity compensation and
gripper normalization. This version has a 50 N linear-gripper force limit and no `gripper_force_limit` constructor argument.
Hardware processes do not need learned models; model services use their own XPolicyLab environments.
These are independent venvs; do not overwrite them with root dependencies using `uv sync`.

## Interface

- Each arm state and target is `[joint1 … joint6, gripper]`: joints in radians; gripper 0 closed, 1 open.
- `end_effector: null` means no additional tool and retains the built-in gripper.
- Construction and `load_model()` do not open CAN; `connect()` creates the i2rt object.
- State copies SDK feedback directly; timestamps are host monotonic times after reading, not motor sampling times.
- `stop()` submits current feedback positions; `close()` waits for service and motor threads before releasing CAN.
  Device references are retained if threads fail to exit, preserving existing shutdown-failure handling.
- TCP is `grasp_site`; FK/IK use each arm's base frame with positions in meters.
- IK fixes `fixed_coordinates={"gripper": opening}` and returns all seven coordinates on success.
- Visualization expands the gripper into two finger joints in the eight-coordinate URDF; hardware commands do not use this expansion.

Component configuration is `manimux/configs/embodiment/arm/yam.yaml`. The robot's
`component_hardware` or local configuration binds CAN channels. For dual-arm assembly, see
[the robot guide](../../robot/yam/README.md).

Teleoperation and demonstration collection are maintained outside this repository.

Raw CAN bypass recording and lock-timing probes were removed. Normal SDK state reads, images, measured joints and
submitted actions remain in episode recordings; historical raw feedback remains available for offline analysis.
