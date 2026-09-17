from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from manimux.embodiments.end_effector import GripperCommand, GripperState
from manimux.embodiments.end_effector.taccap import TacCapGripper


@pytest.fixture
def fixture(monkeypatch):
    clock = NS(now_ns=Mock(return_value=1_000_000_000))
    mapping = NS(valid=True, min_open_rad=1.0, max_open_rad=3.0, reverse=False)
    motor = NS(
        read_status=Mock(return_value=NS(actual_pos=2.0, status=0)), enable=Mock(), disable=Mock()
    )
    device = NS(
        motor=motor,
        position_map=Mock(return_value=mapping),
        set_position=Mock(),
        transport=NS(stop=Mock()),
    )
    endpoint = NS(firmware_sn="SN", side="L", role="F", mcu_device="/dev/fake")
    sdk = NS(
        Side=NS(Left="L", Right="R"),
        Role=NS(Follower="F"),
        scan_grippers=Mock(return_value=[endpoint]),
        FollowerGripper=Mock(return_value=device),
    )
    monkeypatch.setattr(
        "manimux.embodiments.end_effector.taccap.end_effector.importlib.import_module", lambda name: sdk
    )
    driver = TacCapGripper(serial="SN", side="left", kp=8, kd=0.3, clock=clock)
    return driver, device, sdk, clock, mapping


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_contract_rejects_invalid_opening(value):
    with pytest.raises(ValueError):
        GripperCommand(value)
    with pytest.raises(ValueError):
        GripperState(value, 0.0)


def test_connect_feedback_command_and_close(fixture):
    driver, dev, sdk, clock, _ = fixture
    sdk.FollowerGripper.assert_not_called()
    driver.connect()
    driver.connect()
    sdk.FollowerGripper.assert_called_once_with(mcu_device="/dev/fake", open_cameras=False)
    dev.motor.enable.assert_not_called()
    state = driver.get_state()
    assert state == GripperState(0.5, 1.0)
    assert driver.get_state() is state
    driver.send_command(GripperCommand(0.8))
    assert driver.get_state().opening == 0.5  # never substitute target for feedback
    dev.set_position.assert_called_once_with(
        0.8, kp_nm_per_rad=8, kd_nm_s_per_rad=0.3, feedforward_torque_nm=0.0
    )
    dev.motor.enable.assert_called_once()
    driver.stop()
    dev.motor.disable.assert_called_once()
    driver.send_command(GripperCommand(0.2))
    assert dev.motor.enable.call_count == 2
    driver.close()
    driver.close()
    dev.transport.stop.assert_called_once()
    assert dev.motor.disable.call_count == 2
    with pytest.raises(RuntimeError):
        driver.get_state()


def test_reverse_calibration(fixture):
    driver, dev, _, _, mapping = fixture
    mapping.reverse = True
    dev.motor.read_status.return_value.actual_pos = -1.5
    driver.connect()
    assert driver.get_state().opening == 0.25


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), 4.0])
def test_invalid_feedback_is_not_clipped(fixture, raw):
    driver, dev, _, _, _ = fixture
    dev.motor.read_status.return_value.actual_pos = raw
    with pytest.raises(ValueError):
        driver.connect()
    dev.transport.stop.assert_called_once()


def test_read_failure_invalidates_cache_and_stops_command(fixture):
    driver, dev, _, clock, _ = fixture
    driver.connect()
    driver.send_command(GripperCommand(0.5))
    clock.now_ns.return_value = 2_000_000_000
    dev.motor.read_status.side_effect = TimeoutError("read failed")
    with pytest.raises(TimeoutError):
        driver.send_command(GripperCommand(0.8))
    dev.motor.disable.assert_called_once()
    assert dev.set_position.call_count == 1
    with pytest.raises(TimeoutError):
        driver.get_state()


def test_bad_calibration_cleanup(fixture):
    driver, dev, _, _, mapping = fixture
    mapping.valid = False
    with pytest.raises(ValueError):
        driver.connect()
    dev.transport.stop.assert_called_once()
    mapping.valid = True
    driver.connect()
    assert driver.get_state().opening == 0.5


def test_wrong_or_ambiguous_endpoint(fixture):
    driver, _, sdk, _, _ = fixture
    sdk.scan_grippers.return_value *= 2
    with pytest.raises(ValueError):
        driver.connect()
    sdk.FollowerGripper.assert_not_called()


def test_fault_is_not_cleared(fixture):
    driver, dev, _, _, _ = fixture
    dev.motor.read_status.return_value.status = 2
    with pytest.raises(RuntimeError, match="protection"):
        driver.connect()
    dev.motor.enable.assert_not_called()
    dev.transport.stop.assert_called_once()


def test_command_failure_disables(fixture):
    driver, dev, _, _, _ = fixture
    driver.connect()
    dev.set_position.side_effect = RuntimeError("submit failed")
    with pytest.raises(RuntimeError, match="submit"):
        driver.send_command(GripperCommand(0.8))
    dev.motor.disable.assert_called_once()


def test_cleanup_failure_retains_connection_for_retry(fixture):
    driver, dev, _, _, _ = fixture
    driver.connect()
    driver.send_command(GripperCommand(0.5))
    dev.motor.disable.side_effect = RuntimeError("disable failed")
    with pytest.raises(RuntimeError, match="disable"):
        driver.close()
    dev.transport.stop.assert_not_called()
    with pytest.raises(RuntimeError, match="cleanup"):
        driver.connect()
    dev.motor.disable.side_effect = None
    driver.close()
    dev.transport.stop.assert_called_once()


def test_cache_preserves_timestamp_until_new_read(fixture):
    driver, dev, _, clock, _ = fixture
    driver.connect()
    clock.now_ns.return_value = 1_010_000_000
    assert driver.get_state().timestamp == 1.0
    clock.now_ns.return_value = 1_100_000_000
    dev.motor.read_status.return_value.actual_pos = 1.4
    assert driver.get_state().timestamp == 1.1
    assert driver.get_state().opening == pytest.approx(0.2)
