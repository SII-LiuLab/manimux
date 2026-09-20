# RealSense 组件

`RealSenseSensor(SensorBase)` 是唯一的 pyrealsense2 采集实现。相机服务与 YAM
采集调用同一组件；`collection/yam/camera/realsense.py` 只转换 `CameraFrame`。

## 安装

从仓库根目录安装到已有硬件环境：

```bash
uv pip install --python envs/yam/.venv/bin/python -e '.[realsense,collection]'
```

当前参考环境为 Python 3.12、pyrealsense2 2.58.3.10794。SDK 通过环境安装，不复制
到组件目录。设备使用序列号绑定；`get_device_ids()` 只枚举、不重置设备。
构造组件不导入 pyrealsense2，不打开 USB；`start()` 打开 pipeline，`close()` 释放它。

## 帧与参数

- `read()` 返回 RGB uint8 的 `SensorFrame`；序号来自 RGB 帧，时间为主机接收时间。
- `capture()` 额外返回可选 uint16 深度、米/计数比例、同帧 Unix 与单调时间戳。
- `read_with_timestamp()` 提供相机网络服务使用的 RGB/depth/Unix 秒接口。
- `camera_serial` 绑定设备；`width`、`height`、`fps` 显式选择流。
- `enable_depth`、`align_depth`、`flip`、`exposure_us`、`white_balance` 按配置执行。
- 原网络服务默认仍为 640×360、30 Hz、深度对齐；原采集默认仍为 640×480、
  30 Hz、RGB-only，可选的采集深度保持未对齐。新组件 YAML 明确列出选择。

默认 `background: true`，组件在后台采集，网络服务的请求直接读取最新缓存，
不会为每台相机依次等待下一帧。重复读取同一缓存保持相同的时间戳和序号。
YAM 采集通过 `background: false` 交由已有的 CameraWorker 调用 `capture()`，
避免两个线程同时取帧。两条路径共用一个 SDK 实现。
保留首帧等待和 `max_frame_age_sec` 停帧处理；SDK 异常直接交给读取方，
不再像旧网络实现一样在后台自动重启 pipeline，不自动更换设备或分辨率。

配置位于 `manimux/configs/embodiment/sensor/realsense.yaml`。YAM 整机示例中相机序列号在
`manimux/configs/local/yam.example.yaml`；旧相机服务的 `device_id` 字段也在服务入口转换。
这两个配置入口调用同一个组件，不保留第二套 RealSense pipeline 实现。
