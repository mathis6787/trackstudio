"""Dummy single-camera tracker for tests and development."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trackstudio.vision_types import Detection, Track

from .base import BaseTrackerConfig, SingleCameraTracker

logger = logging.getLogger(__name__)


class DummySingleCameraTracker(SingleCameraTracker):
    """Tracker that assigns simple IDs to detections."""

    def __init__(self, config: BaseTrackerConfig | None = None) -> None:
        super().__init__(config or BaseTrackerConfig())
        logger.info("🤖 DummySingleCameraTracker initialized")

    def track(
        self,
        detections: list[Detection],
        camera_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        tracks: list[Track] = []
        for i, detection in enumerate(detections):
            tracks.append(
                Track(
                    track_id=f"cam{camera_id}_track_{i}_{int(timestamp)}",
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    age=1,
                    camera_id=camera_id,
                )
            )
        return tracks

    def get_config_schema(self) -> dict[str, Any]:
        return {}

    def update_config(self, config_update: dict[str, Any]) -> None:
        logger.debug("Dummy tracker has no live configuration to update.")

    def get_statistics(self) -> dict[str, Any]:
        return {"tracker_type": "Dummy"}


# Compatibility alias for older imports inside this repository.
DummyVisionTracker = DummySingleCameraTracker
