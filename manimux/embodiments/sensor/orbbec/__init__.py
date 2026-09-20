"""Orbbec Gemini RGB capture, shared by camera-server consumers."""

from manimux.embodiments.sensor.orbbec.sensor import OrbbecCamera, discover_orbbec

__all__ = ["OrbbecCamera", "discover_orbbec"]
