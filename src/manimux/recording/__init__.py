from manimux.recording.episode import EpisodeRecorder

__all__ = ["EpisodeRecorder"]


def recording_parameters(**options) -> dict:
    """保留记录与视频默认值；不打开文件或启动编码进程。"""

    values = {
        "enabled": True,
        "video_fps": 0.0,
        "video_codec": "mp4v",
        "video_queue_size": 8,
        **options,
    }
    if values["enabled"] is not True:
        raise ValueError("recording cannot be disabled")
    return values
