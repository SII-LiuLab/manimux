"""Shared RealSense capture contracts, using an SDK fake with no USB access."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manimux.embodiments.sensor.realsense import RealSenseSensor


@pytest.fixture
def sdk(monkeypatch):
    events = []
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    depth = np.array([[100, 200]], dtype=np.uint16)
    sensor = SimpleNamespace(set_option=lambda key, value: events.append((key, value)))
    device = SimpleNamespace(
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001),
        query_sensors=lambda: [sensor],
    )

    class Pipeline:
        def __init__(self):
            events.append("create")
            self.sequence = 0

        def start(self, config):
            events.append("start")
            return SimpleNamespace(get_device=lambda: device)

        def wait_for_frames(self, **options):
            self.sequence += 1
            return SimpleNamespace(
                get_color_frame=lambda: SimpleNamespace(
                    get_data=lambda: rgb, get_frame_number=lambda: self.sequence
                ),
                get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth),
            )

        def stop(self):
            events.append("stop")

    rs = SimpleNamespace(
        pipeline=Pipeline,
        config=lambda: SimpleNamespace(
            enable_device=lambda serial: events.append(("serial", serial)),
            enable_stream=lambda *args: events.append(("stream", args)),
        ),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"),
        option=SimpleNamespace(
            enable_auto_exposure="auto_exposure",
            exposure="exposure",
            enable_auto_white_balance="auto_white_balance",
            white_balance="white_balance",
        ),
        align=lambda kind: SimpleNamespace(process=lambda frames: events.append("align") or frames),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    return events, rgb, depth, rs


def test_sensor_lifecycle_does_not_open_during_construction(sdk):
    events, _, _, _ = sdk
    camera = RealSenseSensor(name="wrist", camera_serial="serial", background=False)
    camera.close()
    assert events == []
    camera.start()
    camera.start()
    assert events.count("start") == 1
    assert ("serial", "serial") in events
    camera.close()
    camera.close()
    assert events.count("stop") == 1
    camera.start()
    camera.close()
    assert events.count("start") == events.count("stop") == 2


def test_rgb_depth_frame_and_time_stay_paired(sdk, monkeypatch):
    events, rgb, depth, _ = sdk
    monkeypatch.setattr("manimux.embodiments.sensor.realsense.sensor.time.time", lambda: 12.5)
    camera = RealSenseSensor(
        name="wrist",
        camera_serial="serial",
        enable_depth=True,
        align_depth=True,
        flip=True,
        clock=SimpleNamespace(now_ns=lambda: 42),
        background=False,
    )
    camera.start()
    frame = camera.capture()
    np.testing.assert_array_equal(frame.image, rgb[:, ::-1])
    np.testing.assert_array_equal(frame.depth, depth[:, ::-1])
    assert frame.depth.dtype == np.uint16 and frame.depth_scale == 0.001
    assert (frame.unix_s, frame.monotonic_ns, frame.sequence) == (12.5, 42, 1)
    runtime_frame = camera.read()
    assert (runtime_frame.name, runtime_frame.capture_monotonic_ns, runtime_frame.sequence) == (
        "wrist",
        42,
        2,
    )
    assert events.count("align") == 2
    camera.close()



def test_camera_server_uses_same_component_and_legacy_stream_defaults(sdk, monkeypatch):
    from manimux.servers.camera.server import _open_rgbd

    events, rgb, _, _ = sdk
    # Keep this stream-options test synchronous; cache/thread behavior is tested below.
    monkeypatch.setattr(RealSenseSensor, "_capture_loop", lambda self: None)
    camera = _open_rgbd("front", {"device_id": "serial"})
    assert camera.background is True
    camera.background = False
    assert isinstance(camera, RealSenseSensor)
    image, depth, timestamp = camera.read_with_timestamp()
    np.testing.assert_array_equal(image, rgb)
    assert depth.shape == (1, 2, 1)
    assert ("stream", ("color", 640, 360, "rgb8", 30)) in events
    assert ("stream", ("depth", 640, 360, "z16", 30)) in events
    assert events.count("align") == 1 and timestamp > 0
    camera.close()


def test_background_reads_share_capture_time_and_do_not_wait_for_next_frame(sdk, monkeypatch):
    _, rgb, _, rs = sdk
    blocked = threading.Event()
    release = threading.Event()
    original = rs.pipeline.wait_for_frames

    def next_frame(pipeline, **options):
        if pipeline.sequence == 1:
            blocked.set()
            assert release.wait(timeout=3)
        return original(pipeline, **options)

    monkeypatch.setattr(rs.pipeline, "wait_for_frames", next_frame)
    camera = RealSenseSensor(name="wrist", camera_serial="serial", max_frame_age_sec=3)
    camera.start()
    try:
        assert blocked.wait(timeout=3)
        first = camera.read()
        second = camera.read()
        image, depth, timestamp = camera.read_with_timestamp()
        np.testing.assert_array_equal(image, rgb)
        assert depth is None and timestamp > 0
        assert first.sequence == second.sequence == camera._pipeline.sequence == 1
        assert first.capture_monotonic_ns == second.capture_monotonic_ns
        assert camera.read_with_timestamp()[2] == timestamp
    finally:
        camera._stop_event.set()
        release.set()
        camera.close()
    assert camera._capture_thread is None


def test_background_sdk_failure_reaches_reader(sdk, monkeypatch):
    _, _, _, rs = sdk
    failure = RuntimeError("SDK disconnected")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(rs.pipeline, "wait_for_frames", fail)
    camera = RealSenseSensor(name="wrist", camera_serial="serial")
    camera.start()
    try:
        with pytest.raises(RuntimeError, match="SDK disconnected") as error:
            camera.read()
        assert error.value is failure
    finally:
        camera.close()


def test_startup_setting_failure_releases_pipeline(sdk):
    events, _, _, rs = sdk
    rs.align = lambda _: (_ for _ in ()).throw(RuntimeError("SDK align failure"))
    camera = RealSenseSensor(
        name="wrist", camera_serial="serial", enable_depth=True, align_depth=True
    )
    with pytest.raises(RuntimeError, match="SDK align failure"):
        camera.start()
    camera.close()
    assert events.count("start") == events.count("stop") == 1


def test_yam_camera_service_resolves_assembly_and_local_serials():
    from manimux.servers.camera.server import camera_config
    from manimux.cli import load_config

    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root / "manimux/configs/experiments/put_bottles/pi05/yam_pi05_joint.yaml",
        local=root / "manimux/configs/local/yam.example.yaml",
    )
    cameras = camera_config(config)["sensors"]["cameras"]
    standalone = yaml.safe_load((root / "manimux/configs/embodiment/sensor/cameras/realsense_3_views_standalone.yaml").read_text())["sensors"]["cameras"]
    assert {name: spec["camera_serial"] for name, spec in cameras.items()} == {
        name: spec["device_id"] for name, spec in standalone.items()
    }


def test_network_source_uses_component_lifecycle_without_connecting_at_construction(monkeypatch):
    from manimux.embodiments.sensor.camera_server import driver
    from manimux.embodiments.sensor import SensorBase, build_sensor, sensor_parameters

    events = []
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)

    class Client:
        def __init__(self, *args, **kwargs):
            events.append("connect")

        def ping(self):
            return True

        def get_obs(self, camera_names):
            return {name: rgb for name in camera_names}

        def close(self):
            events.append("close")

    monkeypatch.setattr(driver, "CameraClient", Client)
    source = build_sensor(
        sensor_parameters(
            name="views", driver="camera_server", options={"camera_names": ["top", "wrist"]}
        ),
        SimpleNamespace(now_ns=lambda: 123),
    )
    assert isinstance(source, SensorBase)
    assert events == []
    source.start()
    frames = source.read()
    source.close()
    assert events == ["connect", "close"]
    assert set(frames) == {"top", "wrist"}
    assert all(frame.data is rgb for frame in frames.values())
    assert all(frame.capture_monotonic_ns == 123 and frame.sequence == 1
               for frame in frames.values())


def test_robot_sensor_reads_preserve_single_frames_and_named_bundles():
    from manimux.embodiments.robot import RobotBase
    from manimux.types import SensorFrame

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    top = SensorFrame("device_top", rgb, 11, 1)
    left = SensorFrame("left_wrist", rgb, 22, 2)
    right = SensorFrame("right_wrist", rgb, 33, 3)
    sensors = {
        "top_component": SimpleNamespace(read=lambda: top),
        "network_bundle": SimpleNamespace(read=lambda: {
            "left_wrist": left, "right_wrist": right,
        }),
    }
    robot = SimpleNamespace(
        sensors=sensors, _sensor_started=set(sensors), _lock=threading.RLock()
    )
    frames = RobotBase.read_sensors(robot)
    assert list(frames) == ["top_component", "left_wrist", "right_wrist"]
    assert frames["top_component"] is top
    assert frames["left_wrist"] is left and frames["right_wrist"] is right
    assert [frame.capture_monotonic_ns for frame in frames.values()] == [11, 22, 33]


def test_component_sensor_imports_do_not_load_device_libraries():
    import subprocess

    code = """
import sys
from manimux.embodiments.sensor import SensorBase, build_sensor
from manimux.embodiments.sensor.orbbec import OrbbecCamera
from manimux.embodiments.sensor.realsense import RealSenseSensor
from manimux.embodiments.sensor.taccap import TacCapSensor
assert not {'cv2', 'pyudev', 'pyrealsense2', 'xense.taccap'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
