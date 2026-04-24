"""DeepSORT single-camera tracker."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import supervision as sv
from trackers import DeepSORTTracker

from trackstudio.models.reid_extractor import TorchReIDExtractor
from trackstudio.vision_config import DeepSORTTrackerConfig
from trackstudio.vision_types import Detection, Track

from .base import SingleCameraTracker

logger = logging.getLogger(__name__)


class DeepSORTSingleCameraTracker(SingleCameraTracker):
    """DeepSORT tracker with TorchReID appearance features."""

    def __init__(self, config: DeepSORTTrackerConfig, reid_model: str = "osnet_x1_0") -> None:
        super().__init__(config)
        self.config: DeepSORTTrackerConfig = config
        self.reid_model = reid_model
        self.reid_extractor: TorchReIDExtractor | None = None
        self.trackers: dict[int, Any] = {}
        self._initialize_reid()
        logger.info(f"🎯 DeepSORTSingleCameraTracker initialized with {reid_model}")

    def _initialize_reid(self) -> None:
        from trackstudio.models.reid_singleton import get_reid_extractor  # noqa: PLC0415

        self.reid_extractor = get_reid_extractor(model_name=self.reid_model)
        if self.reid_extractor is None:
            logger.warning("⚠️ ReID extractor could not be initialized - tracking will be less accurate")

    def _get_or_create_tracker(self, camera_id: int) -> Any:
        if camera_id not in self.trackers:
            if self.reid_extractor is None:
                raise RuntimeError("ReID extractor not initialized.")

            self.trackers[camera_id] = DeepSORTTracker(feature_extractor=self.reid_extractor)
            tracker = self.trackers[camera_id]
            self._apply_live_config(tracker)

        return self.trackers[camera_id]

    def _apply_live_config(self, tracker: Any) -> None:
        tc = self.config.tracking
        if hasattr(tracker, "max_age"):
            tracker.max_age = tc.tracker_max_age
        if hasattr(tracker, "min_hits"):
            tracker.min_hits = tc.tracker_min_hits
        if hasattr(tracker, "_metric") and hasattr(tracker._metric, "_metric_params"):
            tracker._metric._metric_params["matching_threshold"] = tc.tracker_matching_threshold

    def track(
        self,
        detections: list[Detection],
        camera_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        if frame is None:
            raise ValueError("Frame is required for DeepSORT.")

        tracker = self._get_or_create_tracker(camera_id)

        if not detections:
            tracker.update(sv.Detections.empty(), frame)
            return []

        sv_detections = sv.Detections(
            xyxy=np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] for d in detections],
                dtype=np.float32,
            ),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in detections], dtype=int),
        )

        tracked_detections = tracker.update(sv_detections, frame)

        tracks: list[Track] = []
        if hasattr(tracked_detections, "tracker_id") and tracked_detections.tracker_id is not None:
            for i, tracker_id in enumerate(tracked_detections.tracker_id):
                if tracker_id is not None and tracker_id >= 0:
                    x1, y1, x2, y2 = tracked_detections.xyxy[i]
                    tracks.append(
                        Track(
                            track_id=f"cam{camera_id}_track_{tracker_id}",
                            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            confidence=float(tracked_detections.confidence[i]),
                            age=1,
                            camera_id=camera_id,
                        )
                    )
        return tracks

    def update_config(self, config_update: dict[str, Any]) -> None:
        new_data = self.config.model_dump()
        for key, value in config_update.items():
            if key in new_data:
                new_data[key] = value

        self.config = DeepSORTTrackerConfig(**new_data)
        for tracker in self.trackers.values():
            self._apply_live_config(tracker)

    def get_reid_features(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray | None:
        if not self.reid_extractor or not tracks:
            return None

        try:
            bboxes = []
            for track in tracks:
                x, y, w, h = track.bbox
                bboxes.append([x, y, x + w, y + h])

            features = self.reid_extractor.extract_features(frame, np.array(bboxes, dtype=np.float32))
            if features is not None:
                logger.debug(f"🔍 Extracted ReID features for {len(tracks)} tracks: shape {features.shape}")
            return features
        except Exception as e:
            logger.error(f"❌ Error extracting ReID features: {e}")
            return None

    def get_config_schema(self) -> dict[str, Any]:
        return self.config.model_json_schema()

    def get_statistics(self) -> dict[str, Any]:
        return {"tracker_type": "DeepSORT"}
