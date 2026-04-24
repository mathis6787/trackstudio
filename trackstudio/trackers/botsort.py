"""BoT-SORT single-camera tracker using BoxMOT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from boxmot.trackers.botsort.botsort import BotSort
except ImportError as e:  # pragma: no cover - exercised through factory tests
    raise ImportError("BoxMOT is required for BoT-SORT. Run `uv sync` to install project dependencies.") from e

from trackstudio.vision_config import BoTSORTTrackerConfig
from trackstudio.vision_types import Detection, Track

from .base import SingleCameraTracker

logger = logging.getLogger(__name__)


class BoxMOTTorchReIDAdapter:
    """Adapter exposing TrackStudio's TorchReID extractor to BoxMOT."""

    def __init__(self, extractor: Any) -> None:
        self.extractor = extractor

    def get_features(self, xyxy_boxes: np.ndarray, frame: np.ndarray) -> np.ndarray:
        features = self.extractor.extract_features(frame, np.asarray(xyxy_boxes, dtype=np.float32))
        if features is None:
            raise RuntimeError("TorchReID returned no features for BoT-SORT.")
        return features


class BoTSORTSingleCameraTracker(SingleCameraTracker):
    """BoT-SORT tracker with optional TorchReID appearance features."""

    def __init__(self, config: BoTSORTTrackerConfig, reid_model: str = "osnet_x1_0") -> None:
        super().__init__(config)
        self.config: BoTSORTTrackerConfig = config
        self.reid_model = reid_model
        self.reid_adapter: BoxMOTTorchReIDAdapter | None = None
        self.trackers: dict[int, Any] = {}
        self._initialize_reid_model()
        logger.info("🎯 BoTSORTSingleCameraTracker initialized")

    def _initialize_reid_model(self) -> None:
        tc = self.config.tracking
        if tc.reid_backend != "trackstudio":
            raise ValueError(f"Unsupported BoT-SORT ReID backend: {tc.reid_backend}")
        if not tc.use_reid_matching:
            self.reid_adapter = None
            return

        from trackstudio.models.reid_singleton import get_reid_extractor  # noqa: PLC0415

        extractor = get_reid_extractor(model_name=self.reid_model)
        if extractor is None:
            raise RuntimeError(
                "BoT-SORT ReID is enabled but TorchReID could not be initialized. "
                "Disable `use_reid_matching` or install/configure TorchReID dependencies."
            )
        self.reid_adapter = BoxMOTTorchReIDAdapter(extractor)

    def _get_or_create_tracker(self, camera_id: int) -> Any:
        if camera_id not in self.trackers:
            tc = self.config.tracking
            if tc.use_reid_matching and self.reid_adapter is None:
                raise RuntimeError("BoT-SORT ReID is enabled but no ReID adapter is available.")
            cmc_method = "sof" if tc.cmc_method in {"none", "sparseOptFlow"} else tc.cmc_method

            self.trackers[camera_id] = BotSort(
                reid_weights=Path(),
                device=self._get_torch_device(),
                half=False,
                track_high_thresh=tc.track_high_thresh,
                track_low_thresh=tc.track_low_thresh,
                new_track_thresh=tc.new_track_thresh,
                track_buffer=tc.track_buffer,
                match_thresh=tc.match_thresh,
                proximity_thresh=tc.proximity_thresh,
                appearance_thresh=tc.appearance_thresh,
                cmc_method=cmc_method,
                frame_rate=tc.frame_rate,
                fuse_first_associate=tc.fuse_first_associate,
                with_reid=False,
            )
            self.trackers[camera_id].with_reid = tc.use_reid_matching
            if tc.cmc_method == "none":
                self.trackers[camera_id].cmc = None
            logger.debug(f"Created BoT-SORT tracker for camera {camera_id}")
        return self.trackers[camera_id]

    def track(
        self,
        detections: list[Detection],
        camera_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        if frame is None:
            raise ValueError("Frame is required for BoT-SORT.")

        tracker = self._get_or_create_tracker(camera_id)
        boxmot_detections = self._detections_to_boxmot(detections)
        embeddings = self._get_detection_embeddings(boxmot_detections, frame)
        tracked = tracker.update(boxmot_detections, frame, embs=embeddings)
        return self._boxmot_to_tracks(tracked, camera_id)

    def update_config(self, config_update: dict[str, Any]) -> None:
        new_data = self.config.model_dump()
        for key, value in config_update.items():
            if key in new_data:
                new_data[key] = value
        self.config = BoTSORTTrackerConfig(**new_data)
        self.trackers.clear()
        self._initialize_reid_model()
        logger.debug("BoT-SORT trackers reset after config update")

    def get_reid_features(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray | None:
        if self.reid_adapter is None or not tracks:
            return None

        bboxes = []
        for track in tracks:
            x, y, w, h = track.bbox
            bboxes.append([x, y, x + w, y + h])

        return self.reid_adapter.get_features(np.asarray(bboxes, dtype=np.float32), frame)

    def get_config_schema(self) -> dict[str, Any]:
        return self.config.model_json_schema()

    def get_statistics(self) -> dict[str, Any]:
        return {"tracker_type": "BoT-SORT"}

    def _get_torch_device(self) -> Any:
        try:
            import torch  # noqa: PLC0415

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"

    def _get_detection_embeddings(self, boxmot_detections: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
        if not self.config.tracking.use_reid_matching:
            return None
        if len(boxmot_detections) == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self.reid_adapter is None:
            raise RuntimeError("BoT-SORT ReID is enabled but no ReID adapter is available.")
        return self.reid_adapter.get_features(boxmot_detections[:, :4], frame)

    def _detections_to_boxmot(self, detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        rows = []
        for detection in detections:
            x, y, w, h = detection.bbox
            rows.append([x, y, x + w, y + h, detection.confidence, detection.class_id])
        return np.asarray(rows, dtype=np.float32)

    def _boxmot_to_tracks(self, tracked: np.ndarray, camera_id: int) -> list[Track]:
        if tracked is None or len(tracked) == 0:
            return []

        tracks: list[Track] = []
        tracked_array = np.asarray(tracked)
        for row in tracked_array:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = row[:4]
            track_id = int(row[4])
            confidence = float(row[5])
            tracks.append(
                Track(
                    track_id=f"cam{camera_id}_track_{track_id}",
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    confidence=confidence,
                    age=1,
                    camera_id=camera_id,
                )
            )
        return tracks
