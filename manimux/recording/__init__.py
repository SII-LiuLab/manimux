import math

from manimux.recording.episode import EpisodeRecorder

__all__ = ["EpisodeRecorder"]


def recording_parameters(**options) -> dict:
    """Preserve recording and video defaults without opening files or starting encoders."""

    values = {
        "enabled": True,
        "video_fps": 0.0,
        "video_codec": "mp4v",
        "video_queue_size": 8,
        "inference_inputs": False,
        "input_queue_size": 2,
        **options,
    }
    if values["enabled"] is not True:
        raise ValueError("recording cannot be disabled")
    if type(values["inference_inputs"]) is not bool:
        raise ValueError("recording.inference_inputs must be a boolean")
    for name in ("video_queue_size", "input_queue_size"):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f"recording.{name} must be a positive integer")
    fps = values["video_fps"]
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps < 0
        or fps > 1e9
    ):
        raise ValueError("recording.video_fps must be finite and between 0 and 1e9")
    if not isinstance(values["video_codec"], str) or len(values["video_codec"]) != 4:
        raise ValueError("recording.video_codec must be a four-character codec")
    return values
