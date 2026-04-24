"""RF-DETR object detector."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from trackstudio.torch_runtime import configure_torch_runtime, is_mps_unsupported_op_error
from trackstudio.vision_config import RFDETRDetectorConfig
from trackstudio.vision_types import Detection

from .base import VisionDetector

configure_torch_runtime()

from rfdetr.detr import RFDETRBase, RFDETRLarge  # noqa: E402

logger = logging.getLogger(__name__)


class RFDETRDetector(VisionDetector):
    """RF-DETR person detector."""

    def __init__(self, config: RFDETRDetectorConfig, model_name: str = "RFDETRBase") -> None:
        super().__init__(config)
        self.config: RFDETRDetectorConfig = config
        self.model_name = model_name
        self.model: RFDETRBase | RFDETRLarge | None = None
        self._forced_cpu_after_mps_error = False
        self._initialize_detector()
        logger.info(f"🎯 RFDETRDetector initialized with {model_name}")

    def _initialize_detector(self) -> None:
        model_kwargs: dict[str, Any] = {"resolution": 560}
        device = os.getenv("TRACKSTUDIO_RFDETR_DEVICE", "").lower()
        if device in {"cpu", "cuda", "mps"}:
            model_kwargs["device"] = device

        if self.model_name == "RFDETRBase":
            self.model = RFDETRBase(**model_kwargs)
        elif self.model_name == "RFDETRLarge":
            self.model = RFDETRLarge(**model_kwargs)
        else:
            model_kwargs["pretrain_weights"] = self.model_name
            self.model = RFDETRBase(**model_kwargs)

    def detect(self, frame: np.ndarray, camera_id: int) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("RF-DETR model not initialized.")

        try:
            sv_detections = self.model.predict(frame)
        except RuntimeError as e:
            if is_mps_unsupported_op_error(e) and self._move_model_to_cpu_after_mps_error():
                logger.warning(
                    "RF-DETR hit an unsupported Apple MPS operator. "
                    "Moved RF-DETR to CPU and retrying this frame."
                )
                sv_detections = self.model.predict(frame)
            else:
                raise

        person_detections: list[Detection] = []

        if hasattr(sv_detections, "class_id") and sv_detections.class_id is not None:
            for i in range(len(sv_detections)):
                if sv_detections.class_id[i] != 1:
                    continue
                if sv_detections.confidence[i] < self.config.detection.confidence_threshold:
                    continue

                x1, y1, x2, y2 = sv_detections.xyxy[i]
                w, h = x2 - x1, y2 - y1

                if (
                    w < self.config.detection.min_box_width
                    or h < self.config.detection.min_box_height
                    or (h > 0 and w / h > self.config.detection.max_aspect_ratio)
                ):
                    continue

                person_detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(w), int(h)),
                        confidence=float(sv_detections.confidence[i]),
                        class_name="person",
                        class_id=1,
                    )
                )

        if len(person_detections) > 1:
            return self._apply_nms(person_detections, self.config.detection.nms_iou_threshold)
        return person_detections

    def update_config(self, config_update: dict[str, Any]) -> None:
        new_data = self.config.model_dump()
        for key, value in config_update.items():
            if key in new_data:
                new_data[key] = value
        self.config = RFDETRDetectorConfig(**new_data)

    def get_config_schema(self) -> dict[str, Any]:
        return self.config.model_json_schema()

    def get_statistics(self) -> dict[str, Any]:
        return {"detector_type": "RF-DETR"}

    def _apply_nms(self, detections: list[Detection], iou_threshold: float) -> list[Detection]:
        return detections

    def _move_model_to_cpu_after_mps_error(self) -> bool:
        if self.model is None or self._forced_cpu_after_mps_error:
            return False

        inner_model = getattr(self.model, "model", None)
        if inner_model is None:
            return False

        try:
            import torch  # noqa: PLC0415

            if hasattr(self.model, "remove_optimized_model"):
                self.model.remove_optimized_model()
            if hasattr(inner_model, "model"):
                inner_model.model = inner_model.model.to("cpu")
            inner_model.device = torch.device("cpu")
            self._forced_cpu_after_mps_error = True
            return True
        except Exception as e:
            logger.error(f"Failed to move RF-DETR to CPU after MPS error: {e}")
            return False
