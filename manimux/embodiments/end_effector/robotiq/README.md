# Robotiq 2F component

Independent `GripperBase` implementation using pymodbus 3.10+ over Modbus RTU.
It is usable without a Franka, RPent or RLinf installation. The assembly supplies
the flange/tool mounting and TCP; the station supplies serial address and measured
open/closed register endpoints. Opening means normalized motor travel, not metres.

Connection only reads status. The gripper must already be activated for execution;
activation, calibration and fault recovery are explicit operator operations.
For an initial communication timeout (0x09), connect polls status for at most
`connect_timeout_s` (default 0.5 s), allowing the device to observe renewed
communication. It performs no writes and succeeds only after the fault clears;
Other faults fail immediately, persistent timeouts fail at the deadline, and
runtime faults are never retried.
Feedback uses actual gPO position, never the requested target or an open/closed
software flag. Every read receives a host monotonic timestamp. Modbus failures,
faults and out-of-calibration feedback propagate. Commands use rACT+rGTO and
configured speed/force; their acknowledgement does not establish arrival.
Optional contact feedback decodes gOBJ only while activated with gGTO set, and
retains gPR as the echoed normalized target. RobotState.grippers carries this
feedback beside actual travel. Contact does not establish target object identity.
Stop clears rGTO, retaining activation and the grasp. The next command resumes;
close stops owned motion then releases serial. Read-only lifecycle performs no
register writes. Arm and gripper submissions are independent, not atomic.

The offline model accepts `visual_model: frames` (default) or `visual_model: 2f_85`
in the assembly's gripper `options`. The 2F-85 display uses the official Robotiq
meshes and articulated URDF, including five mimic joints. Normalized measured
motor travel maps to the upstream nominal knuckle range (0.7929 rad closed,
0 rad open); it is not a calibrated pad-distance measurement. Materials and mesh
detail are retained in binary glTF. Source revisions/hashes and upstream licenses
are under `assets/2f_85/`.

The Franka reference assembly explicitly selects 2F-85. Other models keep the
frame-only visual until a matching resource is supplied. The assembly still owns
its measured mounting and TCP transform; visual selection never changes hardware
commands, register calibration, FK/IK or tool geometry. The standalone URDF omits
the upstream external parent joint and ROS control definitions. No ROS installation
or control driver is required to display it.
