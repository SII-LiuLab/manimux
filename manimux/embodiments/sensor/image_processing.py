"""Opt-in, independent-frame emulation of collection image compression."""
from __future__ import annotations

import io
import time
from copy import deepcopy
from fractions import Fraction

import numpy as np

from manimux.types import SensorFrame


def image_processing_parameters(options: dict) -> dict:
    if not isinstance(options, dict):
        raise ValueError("sensor_reading.image_processing must be a mapping")
    defaults = {
        "type": "jpeg_h264_single_frame", "jpeg_quality": 80,
        "crf": 20, "fps": 30, "gop_size": 30, "thread_count": 2, "workers": 3,
        "preset": "ultrafast", "tune": "zerolatency", "pix_fmt": "yuv420p",
    }
    unknown = options.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown image_processing fields: {sorted(unknown)}")
    values = defaults | options
    for name in ("type", "preset", "tune", "pix_fmt"):
        if values[name] != defaults[name]:
            raise ValueError(f"image_processing.{name} must be {defaults[name]!r}")
    for name, low, high in (("jpeg_quality", 1, 100), ("crf", 0, 51),
                            ("fps", 1, 240), ("gop_size", 1, 1000), ("thread_count", 1, 16),
                            ("workers", 1, 16)):
        if type(values[name]) is not int or not low <= values[name] <= high:
            raise ValueError(f"image_processing.{name} must be an integer in [{low},{high}]")
    return values


class SingleFrameCompression:
    """JPEG round trip, then an independent H.264 I-frame round trip in memory.

    No persistent video state or hardware. The caller caches by source identity.
    This matches the collection's first-frame pipeline, not its inter-frame coding.
    """

    def __init__(self, options: dict, clock):
        self.options = image_processing_parameters(options)
        self.clock = clock
        try:
            import av
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Image compression requires PyAV and OpenCV in the runtime environment "
                "(install manimux[image-processing])"
            ) from error
        # Fail during sensor startup, before the runtime connects robot controllers.
        av.codec.Codec("libx264", "w")
        av.codec.Codec("h264", "r")
        self.av, self.cv2 = av, cv2
        self.versions = {
            "av": av.__version__, "opencv": cv2.__version__,
            "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()},
        }

    def process(self, source: SensorFrame) -> SensorFrame:
        start = time.perf_counter_ns()
        started_ns = self.clock.now_ns()
        rgb = source.data
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or min(rgb.shape[:2]) <= 0 or any(n % 2 for n in rgb.shape[:2])):
            raise ValueError("JPEG/H.264 processing requires even-sized HxWx3 uint8 RGB")
        if (source.camera_meta or {}).get("image_processing") is not None:
            raise ValueError("Image compression was already applied to this source")
        av, cv2, opts = self.av, self.cv2, self.options
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, opts["jpeg_quality"]])
        if not ok:
            raise RuntimeError(f"JPEG encode failed: {source.name}")
        jpeg_encoded = time.perf_counter_ns()
        decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != rgb.shape:
            raise RuntimeError(f"JPEG decode failed: {source.name}")
        jpeg_decoded = time.perf_counter_ns()
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as writer:
            stream = writer.add_stream("libx264", rate=opts["fps"])
            stream.width, stream.height = rgb.shape[1], rgb.shape[0]
            stream.pix_fmt = opts["pix_fmt"]
            stream.options = {"crf": str(opts["crf"]), "preset": opts["preset"], "tune": opts["tune"]}
            stream.codec_context.thread_count = opts["thread_count"]
            stream.codec_context.gop_size = opts["gop_size"]
            frame = av.VideoFrame.from_ndarray(decoded, format="bgr24")
            frame.pts, frame.time_base = 0, Fraction(1, opts["fps"])
            for packet in stream.encode(frame):
                writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        h264_encoded = time.perf_counter_ns()
        with av.open(io.BytesIO(buffer.getvalue()), "r") as reader:
            frames = list(reader.decode(video=0))
            if len(frames) != 1:
                raise RuntimeError("Single-frame H.264 round trip did not return exactly one frame")
            output = frames[0].to_ndarray(format="rgb24")
        output.setflags(write=False)
        finished = time.perf_counter_ns()
        metadata = deepcopy(source.camera_meta or {})
        metadata["image_processing"] = {
            "parameters": dict(opts), "versions": self.versions,
            "semantics": "independent I-frame; no temporal video prediction",
            "started_monotonic_ns": started_ns,
            "completed_monotonic_ns": self.clock.now_ns(),
            "timings_ms": {
                "jpeg_encode_ms": (jpeg_encoded - start) / 1e6,
                "jpeg_decode_ms": (jpeg_decoded - jpeg_encoded) / 1e6,
                "h264_encode_ms": (h264_encoded - jpeg_decoded) / 1e6,
                "h264_decode_ms": (finished - h264_encoded) / 1e6,
                "total_ms": (finished - start) / 1e6,
            },
        }
        return SensorFrame(source.name, output, source.capture_monotonic_ns,
                           source.sequence, depth=source.depth, camera_meta=metadata)
