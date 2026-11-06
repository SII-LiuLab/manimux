"""Explicit-device UVC capture using the existing callback MJPEG/RGB backend.

Constructors are inert. The runtime owns start/close; frames retain host receipt
timestamps and sequence numbers. This opens video only, never a gripper link.
"""

from pathlib import Path

from manimux.embodiments.sensor.taccap.sensor import TacCapCamera, TacCapSensor


class UVCCamera(TacCapCamera):
    def _resolve_device(self) -> Path:
        device = Path(self._camera_serial)
        if not device.is_absolute() or not device.exists():
            raise ValueError(f"UVC device_path must identify an existing absolute video path: {device}")
        return device


class UVCSensor(TacCapSensor):
    _camera_type = UVCCamera

    def __init__(self, *, name, device_path=None, clock=None, **options):
        super().__init__(name=name, camera_serial=device_path, clock=clock, **options)

    def start(self):
        if not self._serial:
            raise ValueError("UVC device_path is required; bind the sensor's station component")
        super().start()


def build_sensor(config, clock):
    return UVCSensor(name=config["name"], clock=clock, **config["options"])
