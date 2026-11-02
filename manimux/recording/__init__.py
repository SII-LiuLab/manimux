"""Recording configuration stays usable without the storage runtime installed."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimux.recording.episode import EpisodeRecorder

__all__ = ["EpisodeRecorder"]


def __getattr__(name: str):
    if name == "EpisodeRecorder":
        from manimux.recording.episode import EpisodeRecorder

        return EpisodeRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def recording_parameters(**options) -> dict:
    """Preserve recording and video defaults without opening files or starting encoders."""
    from manimux.config_validation import merge_parameters

    values = merge_parameters({
        "enabled": True,
        "video_fps": 0.0,
        "video_codec": "mp4v",
        "video_queue_size": 8,
    }, options, 'recording', extra=())
    if values["enabled"] is not True:
        raise ValueError("recording cannot be disabled")
    return values
