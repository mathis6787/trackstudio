"""Factory for complete vision processing components."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from trackstudio.bev import BEVTransformer
from trackstudio.detector_factory import create_detector, get_available_detectors, get_detector_type_from_env
from trackstudio.detectors.base import VisionDetector
from trackstudio.merger_factory import create_merger, get_available_mergers, get_merger_type_from_env
from trackstudio.mergers.base import VisionMerger
from trackstudio.tracker_factory import create_tracker, get_available_trackers, get_tracker_type_from_env
from trackstudio.trackers.base import SingleCameraTracker
from trackstudio.vision_config import VisionSystemConfig

logger = logging.getLogger(__name__)


@dataclass
class VisionComponents:
    """Concrete components for the vision processing pipeline."""

    detector: VisionDetector
    tracker: SingleCameraTracker
    bev_transformer: BEVTransformer
    merger: VisionMerger
    config: VisionSystemConfig


def create_bev_transformer(calibration_file: str | None = None) -> BEVTransformer:
    return BEVTransformer(calibration_file)


def create_vision_system(
    detector_type: str | None = None,
    tracker_type: str | None = None,
    merger_type: str | None = None,
    calibration_file: str | None = None,
) -> VisionComponents:
    """Create detector, tracker, BEV transformer, merger, and system config."""
    detector_type = detector_type or get_detector_type_from_env()
    tracker_type = tracker_type or get_tracker_type_from_env()
    merger_type = merger_type or get_merger_type_from_env()

    available_detectors = get_available_detectors()
    available_trackers = get_available_trackers()
    available_mergers = get_available_mergers()

    if detector_type not in available_detectors:
        raise ValueError(f"Unsupported detector type: {detector_type}. Available detectors: {available_detectors}")
    if tracker_type not in available_trackers:
        raise ValueError(f"Unsupported tracker type: {tracker_type}. Available trackers: {available_trackers}")
    if merger_type not in available_mergers:
        raise ValueError(f"Unsupported merger type: {merger_type}. Available mergers: {available_mergers}")

    try:
        config = VisionSystemConfig(
            detector_type=detector_type,
            tracker_type=tracker_type,
            merger_type=merger_type,
        )  # type: ignore[call-arg]
    except Exception as e:
        logger.error(
            "❌ Failed to create config with detector_type=%s, tracker_type=%s, merger_type=%s: %s",
            detector_type,
            tracker_type,
            merger_type,
            e,
        )
        raise

    detector = create_detector(config)
    tracker = create_tracker(config)
    bev_transformer = create_bev_transformer(calibration_file)
    merger = create_merger(config)
    _optimize_shared_resources(tracker, merger)

    logger.info(
        f"🧠 Vision system created successfully with {detector_type} detector, "
        f"{tracker_type} tracker and {merger_type} merger"
    )
    return VisionComponents(detector, tracker, bev_transformer, merger, config)


def _optimize_shared_resources(tracker: SingleCameraTracker, merger: VisionMerger) -> None:
    """Share expensive resources such as ReID extractors when possible."""
    if hasattr(tracker, "reid_extractor") and hasattr(merger, "reid_extractor"):
        tracker_has_reid = getattr(tracker, "reid_extractor", None) is not None
        merger_has_reid = getattr(merger, "reid_extractor", None) is not None

        if tracker_has_reid and not merger_has_reid:
            merger.reid_extractor = tracker.reid_extractor
            logger.info("♻️ Shared ReID extractor from tracker to merger")
        elif merger_has_reid and not tracker_has_reid:
            tracker.reid_extractor = merger.reid_extractor
            logger.info("♻️ Shared ReID extractor from merger to tracker")


def get_available_vision_systems() -> list[tuple[str, str, str]]:
    detectors = get_available_detectors()
    trackers = get_available_trackers()
    mergers = get_available_mergers()
    return [(detector, tracker, merger) for detector in detectors for tracker in trackers for merger in mergers]


def validate_vision_system_config(detector_type: str, tracker_type: str, merger_type: str) -> bool:
    detector_valid = detector_type in get_available_detectors()
    tracker_valid = tracker_type in get_available_trackers()
    merger_valid = merger_type in get_available_mergers()

    if not detector_valid:
        logger.warning(f"❌ Invalid detector type: {detector_type}")
    if not tracker_valid:
        logger.warning(f"❌ Invalid tracker type: {tracker_type}")
    if not merger_valid:
        logger.warning(f"❌ Invalid merger type: {merger_type}")

    return detector_valid and tracker_valid and merger_valid
