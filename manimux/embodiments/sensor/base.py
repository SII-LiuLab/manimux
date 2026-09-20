"""Explicit sensor lifecycle; reading a cached frame retains its capture metadata."""

from abc import ABC, abstractmethod

from manimux.types import SensorFrame

# 单设备返回一帧；网络订阅可返回同批多路帧，保留各自的时间戳和序号。
SensorRead = SensorFrame | dict[str, SensorFrame]


class SensorBase(ABC):
    """A runtime sensor that is inert until start is explicitly called."""

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> SensorRead:
        """Return one RGB frame or a named bundle, retaining capture timestamps and sequences."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources, including after partial startup; permit retries."""
        raise NotImplementedError
