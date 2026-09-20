"""把公共 RealSense 组件的帧转换为采集格式；硬件实现只保留在组件内。"""

from manimux.embodiments.sensor.realsense import RealSenseSensor

from .interface import CameraFrame, CameraMode


class RealSense:
    mode = CameraMode.MONO

    def __init__(
        self,
        name,
        role,
        width=640,
        height=480,
        fps=30,
        serial=None,
        enable_depth=False,
        exposure_us=None,
        white_balance=None,
    ):
        self.name, self.role = name, role
        self._sensor = RealSenseSensor(
            name=name,
            camera_serial=serial,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
            align_depth=False,
            exposure_us=exposure_us,
            white_balance=white_balance,
            # CameraWorker 已经拥有采集线程，不再让组件创建第二个生产者。
            background=False,
        )
        # 采集 registry 的约定是在构造阶段打开设备，SensorBase 本身仍延迟启动。
        self._sensor.start()

    def image_keys(self):
        return ["rgb"]

    def read(self):
        frame = self._sensor.capture()
        return CameraFrame(
            images={"rgb": frame.image},
            timestamp_ms=frame.unix_s * 1000.0,
            depth=frame.depth,
            depth_scale=frame.depth_scale,
            meta={"sequence": frame.sequence, "monotonic_ns": frame.monotonic_ns},
        )

    def stop(self):
        self._sensor.close()
