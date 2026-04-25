"""Ultralytics YOLO object detector."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from trackstudio.vision_config import YOLODetectorConfig
from trackstudio.vision_types import Detection

from .base import VisionDetector

logger = logging.getLogger(__name__)


class YOLODetector(VisionDetector):
    """YOLO26 axis-aligned box detector for the TrackStudio detector interface."""

    def __init__(self, config: YOLODetectorConfig) -> None:
        super().__init__(config)
        self.config: YOLODetectorConfig = config
        self.model: Any | None = None
        self._loaded_weights: str | None = None
        self._initialize_detector()
        logger.info("🎯 YOLODetector initialized with %s", self.config.model.weights)

    def _initialize_detector(self) -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("Ultralytics is required for detector_type='yolo'. Run `uv sync`.") from e

        self.model = YOLO(self.config.model.weights)
        self._loaded_weights = self.config.model.weights

    def detect(self, frame: np.ndarray, camera_id: int) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("YOLO model not initialized.")

        if self._loaded_weights != self.config.model.weights:
            self._initialize_detector()

        result = self.model.predict(
            frame,
            imgsz=self.config.model.image_size,
            conf=self.config.detection.confidence_threshold,
            iou=self.config.detection.nms_iou_threshold,
            classes=[0] if self.config.model.person_only else None,
            device=self._resolve_device(),
            verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", {}) or {}
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for bbox_xyxy, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
            if confidence < self.config.detection.confidence_threshold:
                continue

            x1, y1, x2, y2 = bbox_xyxy
            w = x2 - x1
            h = y2 - y1

            if (
                w < self.config.detection.min_box_width
                or h < self.config.detection.min_box_height
                or (h > 0 and w / h > self.config.detection.max_aspect_ratio)
            ):
                continue

            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(w), int(h)),
                    confidence=float(confidence),
                    class_name=str(names.get(int(class_id), class_id)),
                    class_id=int(class_id),
                )
            )

        return detections

    def update_config(self, config_update: dict[str, Any]) -> None:
        new_data = self.config.model_dump()
        for key, value in config_update.items():
            if key in new_data:
                new_data[key] = value
        self.config = YOLODetectorConfig(**new_data)

    def get_config_schema(self) -> dict[str, Any]:
        return self.config.model_json_schema()

    def get_statistics(self) -> dict[str, Any]:
        return {"detector_type": f"YOLO26 ({self.config.model.weights})"}

    def _resolve_device(self) -> str:
        device = os.getenv("TRACKSTUDIO_YOLO_DEVICE", self.config.model.device).lower()
        if device in {"cpu", "mps", "cuda"}:
            return device

        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            logger.debug("Could not auto-detect YOLO torch device", exc_info=True)

        return "cpu"
