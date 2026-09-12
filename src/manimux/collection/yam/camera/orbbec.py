"""Gemini 305/335 RGB capture through their UVC color interface.

Select by USB serial AND interface: the first video node is depth, and the
305 also exposes a second color interface. /dev/v4l/by-id can alias those
interfaces, so neither its video-index0 link nor /dev/videoN is a stable RGB
selector. These models expose the primary color stream on interface 04.
"""

from __future__ import annotations

from .base_v4l2 import V4L2Camera
from .interface import CameraFrame, CameraMode

_PRODUCTS = {"0840": "Orbbec Gemini 305", "0800": "Orbbec Gemini 335"}


def discover_orbbec() -> list[dict]:
    """List primary RGB capture nodes without opening or interrupting streams."""
    try:
        import pyudev
    except ImportError as exc:
        raise RuntimeError(
            "Orbbec discovery requires pyudev; install pyudev in the collection Python environment"
        ) from exc

    cameras = []
    for dev in pyudev.Context().list_devices(subsystem="video4linux"):
        if (
            dev.get("ID_VENDOR_ID", "").lower() != "2bc5"
            or dev.get("ID_MODEL_ID", "").lower() not in _PRODUCTS
            or dev.get("ID_USB_INTERFACE_NUM") != "04"
            or ":capture:" not in dev.get("ID_V4L_CAPABILITIES", "")
        ):
            continue
        serial = dev.get("ID_SERIAL_SHORT")
        if serial and dev.device_node:
            cameras.append({
                "kind": "v4l2",
                "serial": serial,
                "product": _PRODUCTS[dev.get("ID_MODEL_ID").lower()],
                "node": dev.device_node,
                "suggested_type": "orbbec",
            })
    return sorted(cameras, key=lambda c: c["serial"])


def find_color_device(serial: str | None) -> dict:
    cameras = discover_orbbec()
    matches = [c for c in cameras if serial is None or c["serial"] == serial]
    if len(matches) != 1:
        available = ", ".join(c["serial"] for c in cameras) or "none"
        raise RuntimeError(
            f"expected one Gemini 305/335 RGB device for serial={serial!r}, "
            f"found {len(matches)} (available serials: {available}); "
            "set a distinct serial for each Orbbec camera"
        )
    return matches[0]


class Orbbec(V4L2Camera):
    mode = CameraMode.MONO

    def __init__(
        self,
        name: str,
        role: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
    ):
        import cv2

        device = find_color_device(serial)
        self.serial = device["serial"]
        super().__init__(name, role, width, height, fps, device=device["node"], fourcc="MJPG")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def image_keys(self) -> list[str]:
        return ["rgb"]

    def read(self) -> CameraFrame:
        rgb, timestamp_ms = self._grab_rgb()
        return CameraFrame(images={"rgb": rgb}, timestamp_ms=timestamp_ms)
