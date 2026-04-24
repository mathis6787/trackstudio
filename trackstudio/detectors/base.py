"""Base classes for object detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from trackstudio.vision_types import Detection


class BaseDetectorConfig(BaseModel):
    """Base configuration class for detector implementations."""

    class Config:
        extra = "allow"


class VisionDetector(ABC):
    """Object detector interface."""

    def __init__(self, config: BaseDetectorConfig) -> None:
        self.config = config

    @abstractmethod
    def detect(self, frame: np.ndarray, camera_id: int) -> list[Detection]:
        """Detect objects in one camera frame."""
        pass

    @abstractmethod
    def update_config(self, config_update: dict[str, Any]) -> None:
        """Update detector configuration."""
        pass

    @abstractmethod
    def get_config_schema(self) -> dict[str, Any]:
        """Return the detector configuration JSON schema."""
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Return detector runtime statistics."""
        pass
