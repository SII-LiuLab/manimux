# Orbbec Gemini RGB

`OrbbecSensor` implements the shared `SensorBase` lifecycle. Construction saves
settings without importing OpenCV or accessing devices. `start()` opens the UVC
stream; `read()` returns a copied RGB `SensorFrame` with its original host capture
time and sequence. `close()` stops capture and releases the device.

Gemini 305/335 selection uses USB serial and UVC interface 04, never `/dev/videoN`
enumeration order. This driver supports RGB only. Install `opencv-python` and
`pyudev` in the camera service environment. Component settings are defined in
`manimux/configs/embodiment/sensor/orbbec.yaml`; bind `camera_serial` in the local
station file. Standalone camera service files also accept legacy `device_id`.
