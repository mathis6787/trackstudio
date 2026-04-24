"""Single-camera tracker interfaces and registry."""

from __future__ import annotations

import logging

from trackstudio.vision_types import BEVTrack, Detection, Track, VisionResult

from .base import BaseTrackerConfig, SingleCameraTracker, VisionTracker

logger = logging.getLogger(__name__)


class TrackerRegistry:
    """Lazy registry for built-in single-camera trackers."""

    def __init__(self) -> None:
        self._trackers: dict[str, type[SingleCameraTracker] | None] = {
            "deepsort": None,
            "bytetrack": None,
            "dummy": None,
        }

    def register(self, name: str, tracker_class: type[SingleCameraTracker]) -> None:
        if not issubclass(tracker_class, SingleCameraTracker):
            raise ValueError(f"{tracker_class} must inherit from SingleCameraTracker")
        self._trackers[name] = tracker_class
        logger.info(f"Registered tracker: {name}")

    def get(self, name: str) -> type[SingleCameraTracker]:
        if name not in self._trackers:
            raise ValueError(f"Unknown tracker: {name}. Available: {list(self._trackers.keys())}")

        tracker_class = self._trackers[name]
        if tracker_class is None:
            if name == "deepsort":
                from .deepsort import DeepSORTSingleCameraTracker  # noqa: PLC0415

                tracker_class = DeepSORTSingleCameraTracker
            elif name == "bytetrack":
                from .bytetrack import ByteTrackSingleCameraTracker  # noqa: PLC0415

                tracker_class = ByteTrackSingleCameraTracker
            elif name == "dummy":
                from .dummy import DummySingleCameraTracker  # noqa: PLC0415

                tracker_class = DummySingleCameraTracker
            self._trackers[name] = tracker_class

        return tracker_class

    def create(self, name: str, **kwargs) -> SingleCameraTracker:
        return self.get(name)(**kwargs)

    def list_available(self) -> list[str]:
        return list(self._trackers.keys())


tracker_registry = TrackerRegistry()

__all__ = [
    "BaseTrackerConfig",
    "SingleCameraTracker",
    "VisionTracker",
    "VisionResult",
    "Detection",
    "Track",
    "BEVTrack",
    "tracker_registry",
    "DeepSORTSingleCameraTracker",
    "ByteTrackSingleCameraTracker",
    "DummySingleCameraTracker",
]


def __getattr__(name: str):
    if name == "DeepSORTSingleCameraTracker":
        from .deepsort import DeepSORTSingleCameraTracker  # noqa: PLC0415

        return DeepSORTSingleCameraTracker
    if name == "ByteTrackSingleCameraTracker":
        from .bytetrack import ByteTrackSingleCameraTracker  # noqa: PLC0415

        return ByteTrackSingleCameraTracker
    if name == "DummySingleCameraTracker":
        from .dummy import DummySingleCameraTracker  # noqa: PLC0415

        return DummySingleCameraTracker
    raise AttributeError(name)
