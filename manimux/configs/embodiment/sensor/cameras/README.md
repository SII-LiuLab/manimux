# 相机服务配置

这些文件定义一次相机服务打开的设备组合；单个设备的组件声明在上一级目录。
历史配置按真实采集差异保留，不再分散到模型目录。

| 文件 | 设备组合 |
| --- | --- |
| `yam.yaml` | 左腕、前方、右腕 RealSense，RGB，当前 Pi05 示例 |
| `yam_rgb.yaml` | 同一组 RGB，显式声明历史帧龄参数 |
| `yam_rgbd.yaml` | 同一组设备，沿用历史配置的默认深度开关 |
| `yam_gemini305.yaml` / `yam_gemini335.yaml` | 两个腕部相机与所选 Gemini |
| `yam_gemini_pair.yaml` | 两个 Gemini |
| `tianji_taccap.yaml` | 两个 TacCap 腕部相机 |

各文件保留原来的设备名、序列号、分辨率和帧率。新工位可以使用实验的
`camera_server` 组件选择和 `--local` 设备绑定，避免复制设备地址到多个实验。
