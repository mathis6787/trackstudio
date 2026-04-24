"""Factory for single-camera tracker components."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from trackstudio.config_registry import get_tracker_names
from trackstudio.trackers.base import SingleCameraTracker
from trackstudio.trackers.dummy import DummySingleCameraTracker
from trackstudio.vision_config import VisionSystemConfig

logger = logging.getLogger(__name__)


TRACKER_REGISTRY: dict[str, type[SingleCameraTracker]] = {
    "dummy": DummySingleCameraTracker,
}


def create_tracker(config: VisionSystemConfig) -> SingleCameraTracker:
    tracker_type = config.tracker_type
    tracker_config = config.get_tracker_config()

    logger.info(f"🏭 Creating tracker of type: {tracker_type}")

    if tracker_type == "dummy":
        return DummySingleCameraTracker(config=tracker_config)

    if tracker_type == "deepsort":
        try:
            from trackstudio.trackers.deepsort import DeepSORTSingleCameraTracker  # noqa: PLC0415

            return DeepSORTSingleCameraTracker(config=tracker_config)
        except ImportError as e:
            logger.error(f"❌ Failed to import DeepSORTSingleCameraTracker: {e}")
            raise ImportError(f"DeepSORT dependencies not available: {e}") from e

    if tracker_type == "bytetrack":
        try:
            from trackstudio.trackers.bytetrack import ByteTrackSingleCameraTracker  # noqa: PLC0415

            return ByteTrackSingleCameraTracker(config=tracker_config)
        except ImportError as e:
            logger.error(f"❌ Failed to import ByteTrackSingleCameraTracker: {e}")
            raise ImportError(f"ByteTrack dependencies not available: {e}") from e

    if tracker_type in TRACKER_REGISTRY:
        tracker_class = TRACKER_REGISTRY[tracker_type]
        return tracker_class(config=tracker_config)

    available_trackers = get_tracker_names()
    raise ValueError(f"Unsupported tracker type: {tracker_type}. Available trackers: {available_trackers}")


def get_tracker_type_from_env() -> str:
    env_tracker = os.getenv("VISION_TRACKER_TYPE", "deepsort").lower()
    available_trackers = get_tracker_names()

    if env_tracker in available_trackers:
        return env_tracker
    if "deepsort" in available_trackers:
        return "deepsort"
    return available_trackers[0] if available_trackers else "dummy"


def register_tracker(name: str, tracker_class: type[SingleCameraTracker]) -> None:
    if not issubclass(tracker_class, SingleCameraTracker):
        raise TypeError(f"Tracker class {tracker_class.__name__} must inherit from SingleCameraTracker")

    if name in TRACKER_REGISTRY:
        logger.warning(f"⚠️ Overriding existing tracker registration: {name}")

    TRACKER_REGISTRY[name] = tracker_class
    logger.info(f"✅ Registered tracker: {name} -> {tracker_class.__name__}")


def register_tracker_class(name: str) -> Callable[[type[SingleCameraTracker]], type[SingleCameraTracker]]:
    def decorator(tracker_class: type[SingleCameraTracker]) -> type[SingleCameraTracker]:
        register_tracker(name, tracker_class)
        return tracker_class

    return decorator


def get_available_trackers() -> list[str]:
    registered = get_tracker_names()
    custom = [name for name in TRACKER_REGISTRY if name not in registered]
    return registered + custom


def get_tracker_info(tracker_type: str) -> dict:
    if tracker_type not in get_available_trackers():
        raise ValueError(f"Unknown tracker type: {tracker_type}")
    return {"type": tracker_type, "registered": tracker_type in TRACKER_REGISTRY}
