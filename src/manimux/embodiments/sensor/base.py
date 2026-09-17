"""Explicit sensor lifecycle; reading a cached frame retains its capture metadata."""

from abc import ABC, abstractmethod

from manimux.types import SensorFrame


class SensorBase(ABC):
    """A runtime sensor that is inert until start is explicitly called."""

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> SensorFrame:
        """Return RGB with the timestamp and sequence belonging to that capture."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources, including after partial startup; permit retries."""
        raise NotImplementedError
