from manimux.recording.episode import EpisodeRecorder

__all__ = ["EpisodeRecorder"]


def recording_parameters(**options) -> dict:
    """Preserve recording and video defaults without opening files or starting encoders."""

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
