"""RealSense 的唯一采集实现；网络服务读后台缓存，采集 worker 直接取帧。"""

import threading
import time
from dataclasses import dataclass

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.sensor.base import SensorBase
from manimux.types import SensorFrame


def get_device_ids():
    """只枚举序列号，不重置或打开正在使用的设备。"""
    import pyrealsense2 as rs

    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


@dataclass(frozen=True)
class Capture:
    """一次 RGB/depth 采集；时间均为主机接收时刻，sequence 来自 RGB 帧。"""

    image: np.ndarray
    depth: np.ndarray | None
    depth_scale: float | None
    unix_s: float
    monotonic_ns: int
    sequence: int


class RealSenseSensor(SensorBase):
    """构造不导入 SDK、不打开 USB；start/close 明确管理一条 pipeline。

    read 返回 runtime 使用的 RGB SensorFrame。capture 同时保留可选的原始
    uint16 深度及米/计数比例，供数据采集使用。深度对齐由配置显式选择。
    """

    def __init__(
        self,
        *,
        name,
        camera_serial=None,
        clock=None,
        width=640,
        height=480,
        fps=30,
        enable_depth=False,
        align_depth=False,
        flip=False,
        exposure_us=None,
        white_balance=None,
        warmup_frames=0,
        max_frame_age_sec=0.30,
        read_timeout_ms=1200,
        background=True,
    ):
        self.name = name
        self.camera_serial = camera_serial
        self.clock = clock if clock is not None else SystemClock()
        self.width, self.height, self.fps = width, height, fps
        self.enable_depth, self.align_depth = enable_depth, align_depth
        self.flip = flip
        self.exposure_us, self.white_balance = exposure_us, white_balance
        self.warmup_frames = warmup_frames
        self.max_frame_age_sec = max_frame_age_sec
        self.read_timeout_ms = read_timeout_ms
        self.background = background
        self._pipeline = None
        self._alignment = None
        self._depth_scale = None
        self._lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._latest_frame = None
        self._capture_error = None

    def start(self):
        with self._lock:
            if self._pipeline is not None:
                return
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            # 设备身份由 local 或采集工位配置给出，不选择枚举列表的第一台。
            config.enable_device(self.camera_serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            if self.enable_depth:
                config.enable_stream(
                    rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
                )
            profile = pipeline.start(config)
            try:
                device = profile.get_device()
                self._depth_scale = (
                    device.first_depth_sensor().get_depth_scale() if self.enable_depth else None
                )
                if self.exposure_us is not None or self.white_balance is not None:
                    # 延续采集站的 D405 曝光设置；SDK 设置失败直接传播，不静默忽略。
                    sensor = device.query_sensors()[0]
                    if self.exposure_us is not None:
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(rs.option.exposure, self.exposure_us)
                    if self.white_balance is not None:
                        sensor.set_option(rs.option.enable_auto_white_balance, 0)
                        sensor.set_option(rs.option.white_balance, self.white_balance)
                self._alignment = (
                    rs.align(rs.stream.color) if self.enable_depth and self.align_depth else None
                )
                for _ in range(self.warmup_frames):
                    pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
            except BaseException:
                # 即使启动中途失败，也释放这个尚未交给调用者的 pipeline。
                pipeline.stop()
                raise
            self._pipeline = pipeline
            with self._frame_lock:
                self._latest_frame = None
                self._capture_error = None
                self._frame_ready.clear()
            self._stop_event.clear()
            if self.background:
                self._capture_thread = threading.Thread(
                    target=self._capture_loop, name=f"realsense_{self.name}", daemon=True
                )
                self._capture_thread.start()

    def capture(self):
        """阻塞获取下一帧；不把 worker 的读取时刻写成另一份采集时间。"""
        with self._lock:
            frames = self._pipeline.wait_for_frames(timeout_ms=self.read_timeout_ms)
            if self._alignment is not None:
                frames = self._alignment.process(frames)
            color = frames.get_color_frame()
            image = np.asanyarray(color.get_data()).copy()
            depth = (
                np.asanyarray(frames.get_depth_frame().get_data()).copy()
                if self.enable_depth
                else None
            )
            unix_s, monotonic_ns = time.time(), self.clock.now_ns()
            if self.flip:
                image = image[::-1, ::-1].copy()
                if depth is not None:
                    depth = depth[::-1, ::-1].copy()
            return Capture(
                image, depth, self._depth_scale, unix_s, monotonic_ns, int(color.get_frame_number())
            )

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self.capture()
            except Exception as exc:
                # 把 SDK 原始异常交给读取方，不在后台静默重启设备。
                with self._frame_lock:
                    self._capture_error = exc
                    self._frame_ready.set()
                break
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_ready.set()

    def _read_capture(self):
        if not self.background:
            return self.capture()
        # 保留旧网络相机的首帧等待与停帧处理；请求线程不等待下一次 USB 采集。
        if not self._frame_ready.wait(timeout=1.5):
            raise TimeoutError("Timed out waiting for a RealSense frame")
        with self._frame_lock:
            frame, error = self._latest_frame, self._capture_error
        if error is not None:
            raise error
        if (self.clock.now_ns() - frame.monotonic_ns) / 1e9 > self.max_frame_age_sec:
            raise RuntimeError("RealSense frame is stale; camera may be stalled")
        return frame

    def read(self):
        frame = self._read_capture()
        return SensorFrame(self.name, frame.image, frame.monotonic_ns, frame.sequence)

    def read_with_timestamp(self):
        """相机网络服务的 RGB/depth/Unix 时间接口；与采集共享同一个 capture。"""
        frame = self._read_capture()
        depth = frame.depth[..., None] if frame.depth is not None else None
        return frame.image, depth, frame.unix_s

    def close(self):
        self._stop_event.set()
        # 先等读取退出，再持有 pipeline 锁释放 USB，避免 join 与采集线程互相等待。
        if self._capture_thread is not None:
            self._capture_thread.join()
            self._capture_thread = None
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.stop()
                self._pipeline = None
