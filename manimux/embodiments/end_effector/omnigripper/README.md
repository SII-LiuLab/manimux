# OmniGripper vendor SDK checkout

## Marvin straight-gripper mounting preview

The offline model accepts `options.visual_asset`: `flange_reference` (default)
or `marvin_straight` (the locally installed Marvin CAD). The latter includes
both fingers and their mimic joint; normalized opening is 0 closed / 1 open.
The source `tool0` is a reference frame with zero translation, not a calibrated
fingertip TCP.

From the repository root, launch the RoboGUI mounting preview:

```bash
envs/flexiv/.venv/bin/python -m manimux.robogui.dashboard \
  --robot flexiv_omnigripper --port 8087 \
  --bridge-endpoint tcp://127.0.0.1:5578 \
  --control-endpoint tcp://127.0.0.1:5579
```

This loads the front-mounted Flexiv dual-arm model at the preset's initial pose,
with both grippers open. It constructs no robot controllers. The dashboard's
waiting status is expected without a runtime service. The assembly
`manimux/configs/embodiment/robot/flexiv_omnigripper_dual.yaml` selects the CAD
through `visual_asset: marvin_straight`. The mount seats the tool by -6.31 mm
along flange Z to align the gripper's rear plane with the wrist end face.
A matching +6.31 mm `tcp_transform` translation retains the existing
flange-reference TCP during visual fitting.

Adjust `components.left_omnigripper.mount` and
`components.right_omnigripper.mount`: translation is in metres and XYZ Euler
rotation is in radians, relative to the corresponding arm flange. The mounts
include the -6.31 mm seating offset; the source gripper URDF already includes its own
-90 degree Z rotation. Restart RoboGUI after editing the mounts.

The local calibrated station uses `.local/flexiv/omnigripper_robot.yaml` and
`.local/flexiv/omnigripper_robogui.yaml`; these preserve each arm's calibrated URDF
and recorded Home pose. Update the local assembly's mounts when fitting this
station. Its dashboard can be launched with
`--config .local/flexiv/omnigripper_robogui.yaml --host 127.0.0.1 --port 8086`.

## Vendor examples

The vendor control examples live in
`sdk/KernalMind-OmniGripper-Control-Examples/`, following the local
end-effector SDK layout. The existing repository ignore rule excludes this
checkout from ManiMux commits. It retains its own Git metadata and upstream files.

- Upstream: https://github.com/KLMmotion/KernalMind-OmniGripper-Control-Examples
- Checked-out revision: `833df8635a10d5afb0474563c275721c3d59342e`
- Setup date: 2026-10-27
- Reference: OmniGripper basic-model user manual v1.2, March 2026, section 3.

Paths below are relative to the vendor checkout:

| Connection | Python example | Instructions |
| --- | --- | --- |
| TJFX arm passthrough | `Marvin_Gripper/TJ_FX_ROBOT_CONTRL_SDK-master/SDK_PYTHON/KM_gripper.py` | `README_EN.md` in the same directory |
| Damiao USB-to-CANFD | `USB_Gripper/u2canfdpy/damiao.py` | `USB_Gripper/u2canfdpy/README_EN.md` |

Use the actual paths above; some upstream README examples use older directory
names or different filename capitalization. Run examples from their containing
directory because they use local modules and native libraries.

The manual specifies Python 3.8–3.10 and recommends NumPy 1.23.5 and
Matplotlib 3.6.3. The USB example additionally requires PyUSB; its bundled
x86-64 extensions target CPython 3.8 and 3.10. Use a separate Python 3.10
environment for these vendor examples, such as `envs/omnigripper/.venv`,
instead of installing these dependencies into the existing Flexiv environment.
The local vendor-example environment is now `envs/omnigripper/.venv`, using
Python 3.10.22, NumPy 1.23.5, Matplotlib 3.6.3 and PyUSB 1.3.1. Dependency
consistency and imports of `src.usb_class` / `src.can_value_type` passed on
2026-10-27. This environment is for standalone vendor examples; ManiMux itself
requires Python 3.11 or 3.12 and is not installed into this environment.

From the ManiMux repository root, repeat adapter enumeration without sending
motor commands:

```bash
envs/omnigripper/.venv/bin/python \
  manimux/embodiments/end_effector/omnigripper/sdk/KernalMind-OmniGripper-Control-Examples/USB_Gripper/u2canfdpy/dev_sn.py
```

This script prints the USB serial of each matching `34b7:6877` adapter; no
output means no matching adapter was enumerated. The latest local check found
zero matches. Two `2e88:4603` HDSC CDC devices were present. A user-assisted
unplug/replug comparison identified the gripper adapter as `ttyACM1`, at
`/dev/serial/by-path/pci-0000:80:14.0-usb-0:3:1.0`. The other device is unrelated
to that comparison. Both devices advertise the same serial number, so use the
confirmed physical port path rather than `/dev/serial/by-id`.

PySerial 3.5 is also installed for a read-only serial-bridge compatibility probe
at `.local/omnigripper/probe_serial.py`. It uses the request framing described by
https://github.com/cmjang/DM_Control_Python and sends only parameter-read and
state-query requests to IDs 1 and 2. Compatibility still requires valid replies;
do not substitute this device's USB ID into the USB-CANFD native driver.

The initial probe failed with `Permission denied`. After the user granted a
device ACL, the port opened and serial writes succeeded. At 921600 baud, no
bytes were received from parameter queries for motor IDs 1 and 2, state queries
for IDs 1 through 16, or the reference protocol's adapter handshake. Repeating
the ID 1/2 queries with PySerial's default DTR/RTS settings also received no
bytes. This does not establish adapter-protocol compatibility or CAN delivery,
and does not establish that the gripper is faulty. Power, wiring, adapter
protocol and CAN bitrate remain unverified. No motor-enable, movement,
parameter-write or flash command was sent.

Reports are `.local/omnigripper/probe_result.json`,
`.local/omnigripper/serial_diagnostic.json` and
`.local/omnigripper/default_serial_diagnostic.json`. Repeat the parameter probe
from the repository root:

```bash
envs/omnigripper/.venv/bin/python .local/omnigripper/probe_serial.py \
  --output .local/omnigripper/probe_result.json
```

At the time of this initial setup there was no registered OmniGripper adapter.
Choose the physical connection
and verify the relevant controller address or USB serial and motor IDs before
hardware use. The stock examples enable motors and generate repeated motion;
they are not offline installation checks. No hardware example was executed
during repository setup.

## Two-adapter investigation, 2026-10-28

The operator identified the current setup as two DM-USB2CANFD boards, each
connected to one OmniGripper motor. The earlier unplug comparison identified
only one board; it did not establish the purpose of the second board.

Both devices currently enumerate as `2e88:4603`, `HDSC CDC Device`, with USB
serial `00000000050C` and CDC ACM interfaces. USB port `3-3` maps to `ttyACM1`
and `3-4` maps to `ttyACM0`. These are port identities, not left/right labels.
USB `bcdDevice=0200` is not a verified application firmware version.

After the operator granted access to both serial and USB device nodes:

- Official DM Device SDK v1.1.0 for Linux x86-64 discovered zero supported
  devices. Its USB identification filter does not match these descriptors.
- SLCAN identity/status queries returned no bytes. The documented DM SLCAN
  `S8`/open sequence followed by register-read frames for IDs 1 and 2 also
  returned no bytes; close-channel requests followed each attempt. Acceptance
  of these commands and the resulting bus configuration could not be verified.
- Official legacy serial-bridge framing at 921600 baud returned no bytes for
  motor-ID broadcast or register reads of IDs, firmware versions and CAN rate.
- The operator reported flashing red LEDs. The DM-USB2CANFD manual associates
  flashing red with CAN bus-off, but firmware identity, actual bus settings,
  motor power and wiring still need confirmation.

No motor enable, motion, motor-parameter write or firmware-flash command was
sent. Communication is **not established**; zero replies do not identify a
particular firmware or prove motor failure.

The neighboring `tianji-control/drivers/gripper.py` is a reference for DM4310
MIT control over Marvin end-effector CAN passthrough, not a USB adapter driver.
Its external `tools/km_gripper.py` dependency is absent from this workstation.
The vendor Marvin/ROS examples select motor IDs 1/2 and feedback IDs 0x11/0x12;
these remain reference values until read from the connected motors. The
neighboring project uses 0=open/1=closed, opposite to ManiMux's GripperBase.

Local evidence is retained in `.local/omnigripper/`:
`dual_adapter_diagnostic_summary.json`, `firmware_identity_probe.json`,
`slcan_identity_diagnostic.json`, `slcan_bus_probe.json`,
`second_adapter_probe_result.json` and `dual_serial_register_probe.json`.

## USB2CAN feedback confirmed, 2026-10-28

Two independent USB2CAN adapters now return valid DM4310 register and status
responses. Both motors use CAN ID 0 and Master ID 0, on separate buses. Earlier
probes incorrectly excluded ID 0, so their lack of replies did not establish a
communication fault. Both report MIT mode and PMAX/VMAX/TMAX of 12.5/30/10;
their hardware ports and CAN bus settings are station-specific. The
`transport.Usb2CanTransport` provides register and feedback queries through
pyserial, including ID 0. `OmniGripper` implements `GripperBase`; the dual Flexiv
assembly binds two independently owned USB2CAN ports. Install `.[omnigripper]`
into the hardware Python environment. Read-only feedback has been verified on
both motors. An isolated low-speed open/close check was subsequently performed
on 2026-10-28 as described below; combined arm motion remains unvalidated.

## ManiMux component and integration

Use `configs/embodiment/robot/flexiv_omnigripper_dual.yaml` with
`configs/local/flexiv_omnigripper_example.yaml`. Each group has seven absolute
arm joint angles in radians, followed by normalized motor travel (0 closed,
1 open). The motor's measured position is mapped to that range and saturated
at its endpoints; it is not finger-pad distance. Establish each physical side
using its USB path and measured endpoints, not USB enumeration or the duplicated
serial number. The private station contains port and calibration values.

The default `flange_reference` resource defines its reference point at the arm
flange. The Flexiv assembly selects the articulated `marvin_straight` visual
resource while retaining that flange-reference TCP explicitly. It is **not a physical grasp TCP**. Replace
the mount and tool geometry with calibrated measurements before using TCP pose
policies or collision geometry. The read-only example uses no sensors or model
server and keeps execution disabled. A trained policy requires a matching
eight-coordinate action layout per group.

`connect` reads motor/master IDs, MIT mode, motor scale limits and measured feedback without
enabling; already-enabled or faulted motors are rejected. Read-only close and
stop send no motor command. An executing first command sends enable, confirms
feedback status, submits a hold at the measured position, then starts a
30 Hz rate-limited worker toward the most recent absolute target in the default
`position` mode. The optional `force_position` mode is described below. Submission
does not imply arrival. `max_follow_rad` bounds target error from feedback;
`max_torque_nm` is a measured-torque fault threshold, not force calibration.
Feedback timestamps are host response times, not device acquisition timestamps.
The adapter does not confirm CAN transmission or arrival. A failed command
attempts disable; `stop` and `close` disable torque and verify disabled status.
Torque removal may release a held object. Cleanup failure retains the connection
for retry, and a later command after a successful stop enables again. This
worker faults are surfaced by later state/command calls. The driver has no
independently verified motor watchdog; do not rely on process
termination as an emergency stop. No fault clear, calibration or homing is
automatic. SDK native calls and serial I/O can block; the system provides no
emergency-stop latency guarantee.

The station's 2026-10-28 manual endpoints were approximately -0.045 to 1.240
rad left and -0.056 to 1.225 rad right. The local -0.02/1.10 range is a
provisional conservative working range, not a physically validated control
limit. Confirm independently before authorized motion. The station README and
probe output are evidence of read-only feedback, not task success.

```bash
python -m pip install -e '.[omnigripper]'
python scripts/validation/omnigripper_readonly.py \
  --config manimux/configs/examples/flexiv_omnigripper_readonly.yaml \
  --local <station.yaml>
```

The probe constructs the assembly but connects only the grippers; it refuses an
executing experiment. It does not connect arms, start services or verify motion.

### Isolated motion check, 2026-10-28

At the user's request, each gripper performed one close/open excursion without
constructing or connecting arm controllers. The existing component drove each
USB2CAN motor sequentially with unchanged `kp=3`, `kd=0.2`, conservative
`open_rad=-0.02` / `closed_rad=1.10` endpoints, and a test-only 0.25 rad/s speed
limit. Both closed to about 1.053 rad and reopened to about 0.018 rad, then
confirmed disabled status 0. The final read-only probe reported normalized
opening about 0.966 for both. Peak measured torque in the completion check was
below 0.21 Nm.

The first left close timed out against a 0.025 rad arrival tolerance, stopping
about 0.047 rad short. It was reopened; the right excursion was then checked
with an explicitly recorded 0.06 rad settling band. This demonstrates measured
motion with the current low-stiffness settings, not precise endpoint arrival,
physical full closure, force calibration or combined robot validation.
Local evidence is in `.local/flexiv/gripper_cycle_20261028_214732.json` and
`.local/flexiv/gripper_cycle_completion_20261028_214823.json`. Test speeds and
settling tolerances were not installed as runtime configuration changes.

## Collection-matched force-position control (2026-11-02)

The Pi05 stack-cube experiment selects `control_mode: force_position` through
`robot.options.component_hardware` for both tools. Its independent workers run at
100 Hz; runtime commands update the latest normalized target. The controller is
adapted from `tianji-control/algos/gripper_force_position.py` and
`drivers/usb2can_transport.py`, with no runtime dependency on that checkout.
The default component remains in `position` mode for existing deployments.

Collection settings: kp 8, kd 0.2, reference speed 1.2 rad/s, acceleration
6 rad/s² and grasp torque budget 0.3 Nm. The complete encoded MIT PD request
includes velocity feedforward and a reserve for position, velocity and torque
quantization. This bounds the predicted request at the latest feedback, not
physical fingertip force or torque between samples. The reference starts at the
measured raw position, including positions just beyond the normalized endpoints.
Delayed steps are capped at two worker periods; a blocked jaw does not accumulate
reference error. The measured torque trip remains independently set to 3 Nm.

The private station supplies `expected_sn`, `can_bitrate_code`, `min_rad` and
`max_rad`, alongside the stable port and opening endpoints. Connect verifies
motor/master IDs, MIT mode, optional SN/CAN-rate registers and the 12.5/30/10
scales required by this controller. It rejects enabled motors. Feedback checks
cover finite values, calibrated position bounds, velocity (1.5 rad/s), temperature
(65 C), torque and age (0.2 s). An executing worker caches actual feedback;
repeated reads preserve its original host-receipt timestamp. A worker failure
is surfaced on subsequent state/command calls and attempts owned-motor disable.
The runtime's robot-level error handling stops the assembly; this is not a
hardware-synchronized or independently guaranteed emergency stop.

The CDC transport initializes the adapter without changing motor parameters.
POSIX writes return as soon as the full frame is accepted and continue only an
unwritten suffix after partial writes. CAN errors and serial-write errors are
not retried as missing feedback; `feedback_retries: 1` permits only one retry
of a feedback query. No motor parameters or flash are written.

The current station's two adapters passed concurrent 100 Hz read-only polling
for five seconds, 501 samples per tool. Expected SN 0/1 and CAN-rate code 4
matched, both statuses stayed disabled (0), and temperatures were 27–30 C.
No motor-enable, MIT target, Home or arm connection was issued by this check.
Repeat using the same experiment/station as deployment:

```bash
envs/flexiv/.venv/bin/python scripts/validation/omnigripper_readonly.py \
  --config .local/flexiv/pi05_stack_cube.yaml --duration 5 --hz 100 \
  --output .local/flexiv/pi05_gripper_readonly_result.json
```

The source controller and actual encoded MIT bytes pass free-travel,
blocked-jaw, reversal and randomized-disturbance tests. Real force-position
motion and a combined policy task have not been validated in this integration.
