# Task 参考图库与 Top 蒙版

参考图提前采集；评测时只选择参考图，在左下角看实时 Top 与参考图的叠加，摆齐后直接 Start。

## 独立采集工具

在仓库根目录运行（相机服务需已启动，不需要模型或机器人 runtime）：

```bash
envs/yam/.venv/bin/python -m manimux.viewer.reference_capture \
  --task put_bottles_into_the_bin
```

打开 <http://127.0.0.1:8087>。

1. 选择已有 Task，或填写新名称并点击“创建 Task”。名称可用中文、字母、数字、下划线、短横线。
2. 摆好第一个初始布局，选择 `01`，点击保存。
3. 依次摆放并保存 `02`–`10`，界面显示采集数量及当前编号的已保存图片。
4. 同一编号再次保存会替换旧图。工具拒绝保存超过 0.5 秒未更新的相机画面。

该工具只订阅现有相机服务的 PUB 流，默认 `tcp://127.0.0.1:5556`、`front_camera`，不会打开 RealSense、加载模型或发机械臂指令。

可选参数：`--root` 指定图库目录，`--camera` 指定相机名，`--camera-endpoint` 指定 PUB 地址，`--port` 指定页面端口。若需其他机器访问，显式使用 `--host 0.0.0.0`。

## 评测

重启现有 Viewer 后，在 **参考布局 · Top** 中选择 Task 和参考图编号。若刚采集了新图片，点击“刷新参考图库”。这里没有保存按钮。

左下角 Action chunks 下方显示 **TOP · 布局复现**。参考透明度默认 50%，0 为纯实时，1 为参考层；也可关闭蒙版看纯实时。参考层另外加 20% 白底，使参考物体颜色更浅，可用“参考白底强度”滑条调整；设为 0 就是原始参考颜色。原来的 Top/Left/Right 相机面板保留。叠加画面使用固定的 Viser 图像控件更新，避免每帧重建 HTML 造成闪烁。

按原来的方式 Prepare，保持暂停，照着蒙版摆放，再点击原来的 Start。不会增加确认步骤或更改控制、模型输入和录像。图库中的 Task 是参考图分组；模型的 Task command 仍使用原来的输入框，不会被自动改写。

图像来自 runtime 的 top 画面，YAM 对应 `front_camera`。Prepare 完成后的暂停阶段会持续更新，runtime idle 时没有新的画面。参考和实时图尺寸不一致时，仅显示实时图并提示；相机位置改变后应恢复机位或重新采集。

小窗口（宽度不超过 900px 或高度不超过 720px）将叠加画面放回侧栏，避免与其他浮动面板重叠。

## 文件组织

默认图库是启动目录下的 `data/evaluation_layouts`：

```text
data/evaluation_layouts/
  put_bottles_into_the_bin/
    01.png
    02.png
    ...
    10.png
  another_task/
    01.png
    ...
    10.png
```

每个 Task 提供十个编号；尚未采满时，评测下拉框只列出已保存的图片。也可事先将自己的参考 PNG 按此结构放入目录。没有预置或虚构参考照片，需要操作员实际采集。

自定义图库时，采集工具用 `--root /path/to/library`，Viewer 用 `--reference-root /path/to/library`，两者指向同一目录。旧版单张 `data/viewer/top_reference.png` 不再自动使用；若需保留，可手动复制到某个 Task 的 `01.png`。
