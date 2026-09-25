"""Orbbec Gemini RGB capture, shared by camera-server consumers."""

from manimux.embodiments.sensor.orbbec.sensor import OrbbecSensor, discover_orbbec

__all__ = ["OrbbecSensor", "discover_orbbec"]
