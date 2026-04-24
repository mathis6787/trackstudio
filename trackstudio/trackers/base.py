"""Base classes for single-camera trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from trackstudio.vision_types import BEVTrack, Detection, Track, VisionResult


class BaseTrackerConfig(BaseModel):
    """Base configuration class for tracking implementations."""

    class Config:
        extra = "allow"


class SingleCameraTracker(ABC):
    """Associates detections into stable per-camera tracks."""

    def __init__(self, config: BaseTrackerConfig) -> None:
        self.config = config

    @abstractmethod
    def track(
        self,
        detections: list[Detection],
        camera_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """Track detected objects across frames for one camera."""
        pass

    @abstractmethod
    def get_config_schema(self) -> dict[str, Any]:
        """Return the tracker's configuration JSON schema."""
        pass

    @abstractmethod
    def update_config(self, config_update: dict[str, Any]) -> None:
        """Update tracker configuration."""
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Return tracker runtime statistics."""
        pass

    def get_reid_features(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray | None:
        """Optionally extract appearance features for tracked objects."""
        return None


# Backward-compatible type alias for modules that still import the old name.
VisionTracker = SingleCameraTracker

__all__ = [
    "BaseTrackerConfig",
    "SingleCameraTracker",
    "VisionTracker",
    "Detection",
    "Track",
    "BEVTrack",
    "VisionResult",
]
