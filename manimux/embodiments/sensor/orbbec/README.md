# Orbbec Gemini RGB

`OrbbecSensor` implements the shared `SensorBase` lifecycle. Construction saves
settings without importing OpenCV or accessing devices. `start()` opens the UVC
stream; `read()` returns a copied RGB `SensorFrame` with its original host frame-receipt
time and sequence. `close()` stops capture and releases the device.

Gemini 305/335 selection uses USB serial and UVC interface 04, never `/dev/videoN`
enumeration order. This driver supports RGB only. Install `opencv-python` and
`pyudev` in the camera service environment. Component settings are defined in
`manimux/configs/embodiment/sensor/orbbec.yaml`; bind `camera_serial` in the local
station file. Standalone camera service files also accept legacy `device_id`.

The collection-matched RGB path requests MJPG, 640×480 at 30 FPS, and one V4L2
buffer. BGR is converted to RGB once; `flip: false` preserves the training view.
Receipt time is recorded immediately after a successful frame read, before color
conversion. This is not a hardware exposure timestamp. Repeated reads retain
that timestamp and sequence; missing/stale capture raises rather than refreshing
old frames. Camera devices require exclusive ownership by the selected capture
service; stop the teleoperation preview before starting the ManiMux camera server.

On 2026-11-02, the three-view Pi05 station passed a five-second check through
`OrbbecSensor -> CameraServer -> CameraServerSensorDriver`: 151 distinct frames
per stream, approximately 30 FPS, RGB uint8 480×640×3, maximum observed age
38.2 ms. The temporary validation server released all devices on exit.
