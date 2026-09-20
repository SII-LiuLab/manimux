# Orbbec Gemini RGB

实现位于 `sensor.py`，`camera_server` 直接使用 `OrbbecCamera`。
按 USB serial 和 UVC interface 04 选择 Gemini 305/335 的 RGB 接口；不按
`/dev/videoN` 的枚举顺序选择设备。当前路径不采集深度。

环境需要 OpenCV 和 pyudev，可安装 ManiMux 的 `collection` 可选依赖。
导入模块不访问设备；当前设备 API 在构造 `OrbbecCamera` 时打开 UVC 并启动
采集线程，`read()` 返回缓存的 RGB 图像，`close()` 结束线程并释放设备。
本次迁移保持采集、翻转和帧过期处理不变，没有新增 SDK 包装。
