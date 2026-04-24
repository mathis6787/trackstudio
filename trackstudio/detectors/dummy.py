"""Dummy detector for tests and development."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trackstudio.vision_types import Detection

from .base import BaseDetectorConfig, VisionDetector

logger = logging.getLogger(__name__)


class DummyDetector(VisionDetector):
    """Detector that produces random person detections."""

    def __init__(self, config: BaseDetectorConfig | None = None) -> None:
        super().__init__(config or BaseDetectorConfig())
        logger.info("🤖 DummyDetector initialized")

    def detect(self, frame: np.ndarray, camera_id: int) -> list[Detection]:
        detections: list[Detection] = []
        num_detections = np.random.randint(0, 4)
        h, w = frame.shape[:2]

        for _ in range(num_detections):
            x = np.random.randint(0, max(1, w - 100))
            y = np.random.randint(0, max(1, h - 100))
            width = np.random.randint(50, 150)
            height = np.random.randint(80, 200)
            detections.append(
                Detection(
                    bbox=(int(x), int(y), int(width), int(height)),
                    confidence=float(0.7 + np.random.random() * 0.3),
                    class_name="person",
                    class_id=0,
                )
            )

        return detections

    def update_config(self, config_update: dict[str, Any]) -> None:
        logger.debug("Dummy detector has no live configuration to update.")

    def get_config_schema(self) -> dict[str, Any]:
        return {}

    def get_statistics(self) -> dict[str, Any]:
        return {"detector_type": "Dummy"}
