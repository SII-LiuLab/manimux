"""Set fixture-supported YAM joints to zero with recorded before/after coordinates.

Default: preview reference angles without SDK/CAN access. --apply first checks
ALL selected joints against the historical raw motor angles, then checks each
again immediately before the official save_zero_position() call. It does not
move to those angles: the arms must already be supported at the reference pose.
The historical reference requires unchanged hardware zeros since its capture.
With --current-fixture, the operator declares the currently installed fixture as
the zero pose. Fresh stable readings replace historical targets for this run only;
the historical file remains unchanged and its usage ledger still prevents retries.
Hardware writes additionally require --fixture-supported: the official zero
operation sends a zero-gain/zero-torque command, and cleanup disables motors.
This operation cannot provide continuous motor-supported holding without a fixture.

The new JSONL report records actual before/after readings and old Home targets.
A locked .state.jsonl beside the reference prevents automatically repeating a
requested zero, including interrupted writes. Do not delete it to retry blindly.
Based on official i2rt/motor_config_tool/set_zero.py at commit 5d47b358.
"""

import argparse
import fcntl
import json
import math
import os
import signal
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TOLERANCE_RAD = 0.001
ZERO_TOLERANCE_RAD = 0.001
FIXTURE_SUPPORT_REQUIRED = (
    "Hardware zeroing requires physical fixture support. The official SDK zero "
    "operation sends zero gains/torque and this script disables motors afterward; "
    "it cannot keep an unsupported arm continuously held."
)


def load_reference(path):
    reference = json.loads(Path(path).read_text())
    if (reference.get("coordinate_frame"), reference.get("units")) != ("motor_raw_rad", "rad"):
        raise ValueError("Reference must contain raw motor coordinates in radians")
    if not isinstance(reference.get("reference_id"), str) or not reference["reference_id"]:
        raise ValueError("Reference needs a nonempty reference_id")
    arms = reference.get("arms")
    if not isinstance(arms, dict) or not arms or set(arms) - {"left", "right"}:
        raise ValueError("Reference arms must be left and/or right")
    for arm, entry in arms.items():
        values = entry.get("positions_rad", []) if isinstance(entry, dict) else []
        if (
            not isinstance(values, list)
            or len(values) != 6
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not -12.5 <= v <= 12.5
                for v in values
            )
        ):
            raise ValueError(f"{arm}: reference needs six finite raw motor angles")
        if not isinstance(entry.get("captured_at"), str) or not entry["captured_at"]:
            raise ValueError(f"{arm}: reference needs captured_at")
    return reference


def record(stream, **event):
    """Persist a record before the next motor operation."""
    stream.write(
        json.dumps({"time": datetime.now(UTC).isoformat(), **event}, allow_nan=False) + "\n"
    )
    stream.flush()
    os.fsync(stream.fileno())


def motor_type_for(joint):
    from i2rt.motor_drivers.dm_driver import MotorType

    return MotorType.DM4340 if joint <= 3 else MotorType.DM4310


def read_position(motor, joint, expected_status):
    """Request raw feedback with 0xCC; no enable, torque command, or offset."""
    motor._drain_bus()
    requested_at = time.time()
    frame = motor._send_message_get_response(
        0x7FF, joint, [joint, 0, 0xCC, 0, 0, 0, 0, 0], max_retry=3
    )
    if (
        frame.is_error_frame
        or frame.is_extended_id
        or frame.is_remote_frame
        or frame.arbitration_id != 0x10 + joint
        or len(frame.data) != 8
        or frame.data[0] & 0x0F != joint
        or not math.isfinite(frame.timestamp)
        or frame.timestamp < requested_at
    ):
        raise RuntimeError(f"J{joint}: stale or malformed feedback")
    feedback = motor.parse_recv_message(frame, motor_type_for(joint), ignore_error=True)
    allowed = (expected_status,) if isinstance(expected_status, int) else expected_status
    if int(feedback.error_code, 16) not in allowed or not math.isfinite(feedback.position):
        raise RuntimeError(f"J{joint}: unexpected motor status or position")
    return feedback.position


def matched_position(motor, arm, joint, target, expected_status):
    # Every sample must match; averaging must not hide a transient mismatch.
    for _ in range(3):
        measured = read_position(motor, joint, expected_status)
        if abs(measured - target) > REFERENCE_TOLERANCE_RAD:
            raise RuntimeError(
                f"{arm}:J{joint}: reference mismatch: measured={measured:+.8f}, "
                f"saved={target:+.8f} rad. No move-to-reference is performed."
            )
    return measured


def zero_joints(
    channels,
    joints,
    report_path,
    reference_path,
    *,
    fixture_supported=False,
    current_fixture=False,
):
    if not fixture_supported:
        raise RuntimeError(FIXTURE_SUPPORT_REQUIRED)
    reference_path = Path(reference_path).resolve()
    reference = load_reference(reference_path)
    if not joints or len(set(joints)) != len(joints) or any(j not in range(1, 7) for j in joints):
        raise ValueError("Select unique arm joints 1-6; gripper 7 is excluded")
    if (
        not channels
        or set(channels) - set(reference["arms"])
        or any(not isinstance(c, str) or not c for c in channels.values())
        or len(set(channels.values())) != len(channels)
    ):
        raise ValueError("Select referenced arms with distinct nonempty CAN channels")
    state_path = reference_path.with_suffix(".state.jsonl")
    if Path(report_path).resolve() in (reference_path, state_path):
        raise ValueError("Report must be separate from the reference and its state file")

    # Reserve a new report and lock the reference's usage log BEFORE opening CAN.
    with (
        Path(report_path).open("x") as report,
        state_path.open("a+") as state,
        ExitStack() as stack,
    ):
        fcntl.flock(state.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.seek(0)
        used = {
            (row["arm"], row["joint"])
            for row in (json.loads(line) for line in state)
            if row.get("event") == "zero_requested"
        }
        if any((arm, joint) in used for arm in channels for joint in joints):
            raise RuntimeError(
                f"Selected joint already has a zero request in {state_path}; "
                "inspect the previous report before any recovery. No automatic retry."
            )
        state.seek(0, os.SEEK_END)
        record(
            report,
            event="start",
            reference=reference,
            reference_path=str(reference_path),
            state_path=str(state_path),
            channels=channels,
            joints=joints,
            zero_pose_source="current_fixture" if current_fixture else "historical_reference",
            reference_tolerance_rad=REFERENCE_TOLERANCE_RAD,
            zero_tolerance_rad=ZERO_TOLERANCE_RAD,
        )
        from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface

        class Interface(DMSingleMotorCanInterface):
            def clean_error(self, *args, **kwargs):
                raise RuntimeError("Motor fault: stopped without resetting the error")

        motors, targets = {}, {}
        arm = joint = None
        try:
            # These constructors open sockets only; no motor_on before ALL checks pass.
            for arm, channel in channels.items():
                motor = Interface(
                    channel=channel, bustype="socketcan", control_mode=ControlMode.MIT
                )
                stack.callback(motor.close)
                motors[arm] = motor
                if motor.bus.recv(timeout=0.05) is not None:
                    raise RuntimeError(f"CAN traffic on {channel}; stop other CAN users first")
                for joint in joints:
                    historical = reference["arms"][arm]["positions_rad"][joint - 1]
                    target = (
                        read_position(motor, joint, expected_status=0)
                        if current_fixture
                        else historical
                    )
                    measured = matched_position(motor, arm, joint, target, expected_status=0)
                    targets[arm, joint] = target
                    record(
                        report,
                        event="precheck",
                        arm=arm,
                        joint=joint,
                        reference_rad=target,
                        historical_reference_rad=historical,
                        historical_delta_rad=measured - historical,
                        measured_rad=measured,
                    )
            record(report, event="all_references_matched")

            for arm, motor in motors.items():
                for joint in joints:
                    target = targets[arm, joint]
                    try:
                        motor._drain_bus()
                        motor.motor_on(joint, motor_type_for(joint))
                        before = matched_position(motor, arm, joint, target, expected_status=1)
                        record(
                            report,
                            event="before",
                            arm=arm,
                            joint=joint,
                            reference_rad=target,
                            before_rad=before,
                        )
                        # A request is uncertain until readback, and must never be retried silently.
                        intent = dict(
                            event="zero_requested",
                            reference_id=reference["reference_id"],
                            arm=arm,
                            joint=joint,
                            channel=channels[arm],
                            reference_rad=target,
                            before_rad=before,
                            report_path=str(Path(report_path).resolve()),
                        )
                        record(state, **intent)
                        record(report, **intent)

                        motor.save_zero_position(joint)  # Official hardware-zero operation.

                        after = read_position(motor, joint, expected_status=(0, 1))
                        verified = abs(after) <= ZERO_TOLERANCE_RAD
                        result = dict(
                            event="after",
                            arm=arm,
                            joint=joint,
                            reference_rad=target,
                            before_rad=before,
                            after_rad=after,
                            verified=verified,
                            old_home_in_new_coordinates_rad=after - before,
                        )
                        record(report, **result)
                        record(state, **result)
                        if not verified:
                            raise RuntimeError(f"{arm}:J{joint}: zero readback failed: {after}")
                        print(f"{arm}:J{joint}: {before:+.8f} -> {after:+.8f} rad", flush=True)
                    finally:
                        try:
                            motor._drain_bus()
                            motor.motor_off(joint)
                            read_position(motor, joint, expected_status=0)
                        except BaseException as error:
                            record(
                                report,
                                event="disable_failed",
                                arm=arm,
                                joint=joint,
                                error=str(error),
                            )
                            raise
            record(report, event="complete")
        except BaseException as error:
            record(report, event="failed", arm=arm, joint=joint, error=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=ROOT / "manimux/configs/local/station.yaml")
    parser.add_argument(
        "--reference", type=Path, required=True, help="Saved historical raw-angle JSON"
    )
    parser.add_argument("--arms", nargs="+", choices=("left", "right"), required=True)
    parser.add_argument(
        "--joints", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7))
    )
    parser.add_argument("--report", type=Path, help="New JSONL backup path; required with --apply")
    parser.add_argument(
        "--apply", action="store_true", help="Verify reference pose, then write zero"
    )
    parser.add_argument(
        "--fixture-supported",
        action="store_true",
        help="Selected arms are physically supported throughout zeroing and motor disable",
    )
    parser.add_argument(
        "--current-fixture",
        action="store_true",
        help="Zero the installed fixture pose; retain historical reference for audit only",
    )
    args = parser.parse_args(argv)
    if args.apply and not args.fixture_supported:
        parser.error(FIXTURE_SUPPORT_REQUIRED)
    if len(set(args.arms)) != len(args.arms) or len(set(args.joints)) != len(args.joints):
        parser.error("Select each arm and joint once")
    if args.apply and args.report is None:
        parser.error("--apply requires --report")
    reference = load_reference(args.reference)
    if set(args.arms) - set(reference["arms"]):
        parser.error("Every selected arm must be present in the reference")
    import yaml

    station = yaml.safe_load(args.local.read_text())
    channels = {arm: station["robot"]["components"][f"{arm}_yam"]["channel"] for arm in args.arms}
    if any(not isinstance(c, str) or not c for c in channels.values()) or len(
        set(channels.values())
    ) != len(channels):
        parser.error("Selected arms must have distinct nonempty CAN channels")
    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "preview",
                    "hardware_opened": False,
                    "channels": channels,
                    "joints": args.joints,
                    "reference_id": reference["reference_id"],
                    "reference": {arm: reference["arms"][arm] for arm in args.arms},
                    "reference_tolerance_rad": REFERENCE_TOLERANCE_RAD,
                    "zero_pose_source": (
                        "current_fixture" if args.current_fixture else "historical_reference"
                    ),
                    "operation": (
                        "Capture and verify the current fixture pose, then zero; no motion"
                        if args.current_fixture
                        else "Verify the saved pose before zeroing; no motion to the reference"
                    ),
                    "requires_physical_fixture_support": True,
                    "continuous_motor_hold_during_zero": False,
                    "old_home": "Actual after_rad - before_rad, saved only when applied",
                    "usage_state_path": str(args.reference.resolve().with_suffix(".state.jsonl")),
                },
                indent=2,
            )
        )
        return

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        zero_joints(
            channels,
            args.joints,
            args.report,
            args.reference,
            fixture_supported=args.fixture_supported,
            current_fixture=args.current_fixture,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(f"Saved zeroing readings and old Home targets: {args.report}")


if __name__ == "__main__":
    main()
