"""Factory for object detector components."""

from __future__ import annotations

import logging
import os

from trackstudio.config_registry import get_detector_names
from trackstudio.detectors.base import VisionDetector
from trackstudio.detectors.dummy import DummyDetector
from trackstudio.vision_config import VisionSystemConfig

logger = logging.getLogger(__name__)


def create_detector(config: VisionSystemConfig) -> VisionDetector:
    detector_type = config.detector_type
    detector_config = config.get_detector_config()

    logger.info(f"🏭 Creating detector of type: {detector_type}")

    if detector_type == "dummy":
        return DummyDetector(config=detector_config)

    if detector_type == "rfdetr":
        try:
            from trackstudio.detectors.rfdetr import RFDETRDetector  # noqa: PLC0415

            return RFDETRDetector(config=detector_config)
        except ImportError as e:
            logger.error(f"❌ Failed to import RFDETRDetector: {e}")
            raise ImportError(f"RFDETRDetector dependencies not available: {e}") from e

    available_detectors = get_detector_names()
    raise ValueError(f"Unsupported detector type: {detector_type}. Available detectors: {available_detectors}")


def get_detector_type_from_env() -> str:
    env_detector = os.getenv("VISION_DETECTOR_TYPE", "rfdetr").lower()
    available_detectors = get_detector_names()

    if env_detector in available_detectors:
        return env_detector
    if "rfdetr" in available_detectors:
        return "rfdetr"
    return available_detectors[0] if available_detectors else "dummy"


def get_available_detectors() -> list[str]:
    return get_detector_names()
