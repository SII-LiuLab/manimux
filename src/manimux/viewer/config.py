"""Viewer display mode and optional manual camera sources."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CameraPanelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    top: str = Field(default="top", min_length=1)
    left: str = Field(default="left", min_length=1)
    right: str = Field(default="right", min_length=1)

    def resolve(
        self, available: Collection[str], normalize: Callable[[str], str]
    ) -> dict[str, str]:
        """Resolve canonical defaults via robot aliases; explicit sources match exactly."""
        resolved = {}
        for panel, source in self.model_dump().items():
            if source in available:
                resolved[panel] = source
            elif source in {"top", "left", "right"}:
                match = next((name for name in available if normalize(name) == source), None)
                if match is not None:
                    resolved[panel] = match
        return resolved


class ViewerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_mode: Literal["policy", "manual"] = "policy"
    cameras: CameraPanelsConfig = Field(default_factory=CameraPanelsConfig)


def load_viewer_config(path: Path | None = None) -> ViewerConfig:
    if path is None:
        return ViewerConfig()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ViewerConfig.model_validate({} if payload is None else payload)
