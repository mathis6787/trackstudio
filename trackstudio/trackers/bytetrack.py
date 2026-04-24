"""ByteTrack single-camera tracker."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from trackstudio.vision_config import ByteTrackTrackerConfig
from trackstudio.vision_types import Detection, Track

from .base import SingleCameraTracker

logger = logging.getLogger(__name__)


class ByteTrackSingleCameraTracker(SingleCameraTracker):
    """ByteTrack tracker using Kalman filtering and IoU association."""

    def __init__(self, config: ByteTrackTrackerConfig) -> None:
        super().__init__(config)
        self.config: ByteTrackTrackerConfig = config
        self.trackers: dict[int, ByteTrack] = {}
        logger.info("🎯 ByteTrackSingleCameraTracker initialized")

    def _get_or_create_tracker(self, camera_id: int) -> ByteTrack:
        if camera_id not in self.trackers:
            tc = self.config.tracking
            self.trackers[camera_id] = ByteTrack(
                track_activation_threshold=tc.track_activation_threshold,
                lost_track_buffer=tc.lost_track_buffer,
                minimum_matching_threshold=tc.minimum_matching_threshold,
                frame_rate=tc.frame_rate,
                minimum_consecutive_frames=tc.minimum_consecutive_frames,
            )
            logger.debug(f"Created ByteTrack for camera {camera_id}")
        return self.trackers[camera_id]

    def track(
        self,
        detections: list[Detection],
        camera_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        tracker = self._get_or_create_tracker(camera_id)

        if not detections:
            tracker.update_with_detections(sv.Detections.empty())
            return []

        sv_detections = sv.Detections(
            xyxy=np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]] for d in detections],
                dtype=np.float32,
            ),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in detections], dtype=int),
        )

        tracked = tracker.update_with_detections(sv_detections)

        tracks: list[Track] = []
        if hasattr(tracked, "tracker_id") and tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                if tid is None or tid < 0:
                    continue
                x1, y1, x2, y2 = tracked.xyxy[i]
                tracks.append(
                    Track(
                        track_id=f"cam{camera_id}_track_{tid}",
                        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        confidence=float(tracked.confidence[i]),
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
        self.config = ByteTrackTrackerConfig(**new_data)
        self.trackers.clear()
        logger.debug("ByteTrack trackers reset after config update")

    def get_config_schema(self) -> dict[str, Any]:
        return self.config.model_json_schema()

    def get_statistics(self) -> dict[str, Any]:
        return {"tracker_type": "ByteTrack"}
