"""Vision system configuration models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, validator

from trackstudio.config_registry import register_detector_config, register_merger_config, register_tracker_config
from trackstudio.detectors.base import BaseDetectorConfig
from trackstudio.trackers.base import BaseTrackerConfig


def slider_field(default: float, min_val: float, max_val: float, step: float, title: str, description: str) -> float:
    return Field(
        default=default,
        title=title,
        description=description,
        json_schema_extra={"ui_control": "slider", "min": min_val, "max": max_val, "step": step, "type": "float"},
    )


def int_slider_field(default: int, min_val: int, max_val: int, step: int, title: str, description: str) -> int:
    return Field(
        default=default,
        title=title,
        description=description,
        json_schema_extra={"ui_control": "slider", "min": min_val, "max": max_val, "step": step, "type": "integer"},
    )


def bool_field(default: bool, title: str, description: str) -> bool:
    return Field(
        default=default,
        title=title,
        description=description,
        json_schema_extra={"ui_control": "toggle", "type": "boolean"},
    )


def select_field(default: str, options: list[str], title: str, description: str) -> str:
    return Field(
        default=default,
        title=title,
        description=description,
        json_schema_extra={"ui_control": "select", "options": options, "type": "string"},
    )


class DetectionConfig(BaseModel):
    """Configuration for object detection filtering."""

    confidence_threshold: float = slider_field(
        0.25, 0.1, 0.95, 0.05, "Confidence Threshold", "Minimum confidence for a detection."
    )
    nms_iou_threshold: float = slider_field(
        0.5, 0.1, 1.0, 0.05, "NMS IoU Threshold", "IoU threshold for Non-Maximum Suppression."
    )
    min_box_width: int = int_slider_field(15, 1, 200, 1, "Min Box Width", "Minimum width of a bounding box.")
    min_box_height: int = int_slider_field(30, 1, 400, 1, "Min Box Height", "Minimum height of a bounding box.")
    max_aspect_ratio: float = slider_field(4.0, 1.0, 10.0, 0.1, "Max Aspect Ratio", "Maximum aspect ratio (w/h).")


class YOLOModelConfig(BaseModel):
    """Configuration for Ultralytics YOLO detection models."""

    weights: str = select_field(
        "yolo26n.pt",
        ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        "Weights",
        "YOLO26 detection checkpoint. Larger models are usually more accurate and slower.",
    )
    image_size: int = int_slider_field(
        640, 320, 1280, 32, "Image Size", "Inference image size passed to Ultralytics YOLO."
    )
    device: str = select_field(
        "auto",
        ["auto", "cpu", "mps", "cuda"],
        "Device",
        "Inference device. Auto prefers CUDA, then Apple MPS, then CPU.",
    )
    person_only: bool = bool_field(
        True,
        "Person Only",
        "Only emit COCO person detections. Disable to emit all YOLO classes.",
    )


@register_detector_config("rfdetr")
class RFDETRDetectorConfig(BaseDetectorConfig):
    """Configuration for RF-DETR object detection."""

    detection: DetectionConfig = Field(default_factory=DetectionConfig, title="Detection Parameters")


@register_detector_config("yolo")
class YOLODetectorConfig(BaseDetectorConfig):
    """Configuration for Ultralytics YOLO object detection."""

    model: YOLOModelConfig = Field(default_factory=YOLOModelConfig, title="YOLO26 Model")
    detection: DetectionConfig = Field(default_factory=DetectionConfig, title="Detection Parameters")


@register_detector_config("dummy")
class DummyDetectorConfig(BaseDetectorConfig):
    """Configuration for dummy detection."""

    pass


class DeepSORTConfig(BaseModel):
    """Configuration for DeepSORT single-camera tracking."""

    tracker_max_age: int = int_slider_field(30, 5, 200, 1, "Max Track Age", "Frames to keep a track without detection.")
    tracker_min_hits: int = int_slider_field(1, 1, 10, 1, "Min Hits to Start", "Consecutive hits to start a track.")
    tracker_max_iou_distance: float = slider_field(
        0.7, 0.1, 1.0, 0.05, "Max IoU Distance", "Max IoU distance for track association."
    )
    tracker_max_cosine_distance: float = slider_field(
        0.3, 0.1, 1.0, 0.05, "Max Cosine Distance", "Max cosine distance for appearance."
    )
    tracker_matching_threshold: float = slider_field(
        0.3, 0.1, 1.0, 0.05, "ReID Matching Threshold", "ReID feature matching threshold."
    )


@register_tracker_config("deepsort")
class DeepSORTTrackerConfig(BaseTrackerConfig):
    """Configuration for DeepSORT tracking."""

    tracking: DeepSORTConfig = Field(default_factory=DeepSORTConfig, title="DeepSORT Parameters")


class ByteTrackConfig(BaseModel):
    """Configuration for ByteTrack single-camera tracking."""

    track_activation_threshold: float = slider_field(
        0.25,
        0.05,
        0.95,
        0.05,
        "Activation Threshold",
        "Min confidence to START a new track. Re-association ignores this threshold.",
    )
    lost_track_buffer: int = int_slider_field(
        50,
        5,
        300,
        5,
        "Lost Track Buffer (frames)",
        "Frames to keep a track alive without a matching detection.",
    )
    minimum_matching_threshold: float = slider_field(
        0.8,
        0.1,
        1.0,
        0.05,
        "IoU Matching Threshold",
        "IoU threshold for the primary matching stage. Lower = more lenient.",
    )
    frame_rate: int = int_slider_field(
        10,
        1,
        60,
        1,
        "Frame Rate",
        "FPS used by ByteTrack's Kalman predictor. Match your vision_fps.",
    )
    minimum_consecutive_frames: int = int_slider_field(
        1,
        1,
        10,
        1,
        "Min Consecutive Frames",
        "Frames a track must appear before being reported.",
    )


@register_tracker_config("bytetrack")
class ByteTrackTrackerConfig(BaseTrackerConfig):
    """Configuration for ByteTrack tracking."""

    tracking: ByteTrackConfig = Field(default_factory=ByteTrackConfig, title="ByteTrack Parameters")


class BoTSORTConfig(BaseModel):
    """Configuration for BoT-SORT single-camera tracking."""

    model_config = ConfigDict(extra="forbid")

    track_high_thresh: float = slider_field(
        0.5, 0.05, 0.95, 0.05, "High Confidence Threshold", "Detection confidence threshold for first association."
    )
    track_low_thresh: float = slider_field(
        0.1, 0.01, 0.5, 0.01, "Low Confidence Threshold", "Lower confidence bound for second-stage candidate detections."
    )
    new_track_thresh: float = slider_field(
        0.6, 0.05, 0.95, 0.05, "New Track Threshold", "Confidence required to initialize a new track."
    )
    track_buffer: int = int_slider_field(
        50, 5, 300, 5, "Lost Track Buffer (frames)", "Frames to keep an unmatched track alive."
    )
    match_thresh: float = slider_field(
        0.8, 0.1, 1.0, 0.05, "Matching Threshold", "Association threshold for matching tracks to detections."
    )
    proximity_thresh: float = slider_field(
        0.5, 0.1, 1.0, 0.05, "Proximity Threshold", "IoU gate used before appearance matching."
    )
    appearance_thresh: float = slider_field(
        0.25, 0.05, 1.0, 0.05, "Appearance Threshold", "Maximum embedding distance accepted for ReID matching."
    )
    use_reid_matching: bool = bool_field(
        True,
        "Use ReID Matching",
        "Use TrackStudio ReID embeddings for BoT-SORT appearance association.",
    )
    reid_backend: str = select_field(
        "trackstudio",
        ["trackstudio"],
        "ReID Backend",
        "Source of ReID embeddings. TrackStudio uses its shared TorchReID/OSNet extractor.",
    )
    cmc_method: str = select_field(
        "none",
        ["none", "ecc", "orb", "sof", "sift", "sparseOptFlow"],
        "Camera Motion Compensation",
        "Camera motion compensation method. Use none for fixed cameras.",
    )
    frame_rate: int = int_slider_field(
        10, 1, 60, 1, "Frame Rate", "FPS used by BoT-SORT's track buffer scaling. Match your vision_fps."
    )
    fuse_first_associate: bool = bool_field(
        False, "Fuse First Association", "Fuse motion and appearance in the first association step."
    )


@register_tracker_config("botsort")
class BoTSORTTrackerConfig(BaseTrackerConfig):
    """Configuration for BoT-SORT tracking."""

    tracking: BoTSORTConfig = Field(default_factory=BoTSORTConfig, title="BoT-SORT Parameters")


@register_tracker_config("dummy")
class DummyTrackerConfig(BaseTrackerConfig):
    """Configuration for dummy tracking."""

    pass


@register_merger_config("bev_cluster")
class CrossCameraConfig(BaseModel):
    """Configuration for BEV cluster cross-camera merging."""

    spatial_threshold: float = slider_field(
        50.0,
        10.0,
        200.0,
        5.0,
        "Spatial Threshold (px)",
        "Max distance in BEV pixels to consider two tracks for merging.",
    )
    appearance_threshold: float = slider_field(
        0.4, 0.1, 1.0, 0.05, "Appearance Threshold", "Max feature distance for appearance matching. Lower is stricter."
    )
    max_track_age_s: float = slider_field(
        3.0, 1.0, 10.0, 0.5, "Max Track Age (s)", "Seconds to keep a global track without updates."
    )
    appearance_weight: float = slider_field(
        0.3, 0.0, 1.0, 0.05, "Appearance Weight", "Weight of appearance vs. spatial distance in matching (0-1)."
    )
    smoothing_alpha: float = slider_field(
        0.3, 0.0, 1.0, 0.05, "Position Smoothing", "Alpha for exponential smoothing of position. Lower is more smooth."
    )
    velocity_alpha: float = slider_field(
        0.5, 0.0, 1.0, 0.05, "Velocity Smoothing", "Alpha for smoothing velocity. Lower is more smooth."
    )


def _create_config_system() -> tuple[type[BaseModel], str, str, str]:
    from trackstudio.config_registry import get_config_classes  # noqa: PLC0415

    try:
        VisionSystemConfig, DetectorType, TrackerType, MergerType = get_config_classes()  # noqa: N806
        return VisionSystemConfig, DetectorType, TrackerType, MergerType
    except Exception as e:
        import logging  # noqa: PLC0415

        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️ Failed to get dynamic config classes, using fallback: {e}")

        class BasicVisionSystemConfig(BaseModel):
            detector_type: str = "rfdetr"
            tracker_type: str = "deepsort"
            merger_type: str = "bev_cluster"

            def get_detector_config(self) -> BaseDetectorConfig:
                from trackstudio.config_registry import get_registered_detector_configs  # noqa: PLC0415

                configs = get_registered_detector_configs()
                if self.detector_type in configs:
                    return configs[self.detector_type]()
                return BaseDetectorConfig()

            def get_tracker_config(self) -> BaseTrackerConfig:
                from trackstudio.config_registry import get_registered_tracker_configs  # noqa: PLC0415

                configs = get_registered_tracker_configs()
                if self.tracker_type in configs:
                    return configs[self.tracker_type]()
                return BaseTrackerConfig()

            def get_merger_config(self) -> BaseModel:
                from trackstudio.config_registry import get_registered_merger_configs  # noqa: PLC0415

                configs = get_registered_merger_configs()
                if self.merger_type in configs:
                    return configs[self.merger_type]()
                return CrossCameraConfig()

            def get_available_detectors(self) -> list[str]:
                return ["rfdetr", "yolo", "dummy"]

            def get_available_trackers(self) -> list[str]:
                return ["deepsort", "bytetrack", "botsort", "dummy"]

            def get_available_mergers(self) -> list[str]:
                return ["bev_cluster"]

        return BasicVisionSystemConfig, str, str, str


_VisionSystemConfig: type[BaseModel] | None = None
_DetectorType: str | None = None
_TrackerType: str | None = None
_MergerType: str | None = None


def get_vision_system_config(force_refresh: bool = False) -> type[BaseModel]:
    global _VisionSystemConfig, _DetectorType, _TrackerType, _MergerType  # noqa: PLW0603

    if _VisionSystemConfig is None or force_refresh:
        try:
            _VisionSystemConfig, _DetectorType, _TrackerType, _MergerType = _create_config_system()
        except Exception as e:
            import logging  # noqa: PLC0415

            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Could not create dynamic config system: {e}, using fallback")

            class _VisionSystemConfigFallback(BaseModel):
                detector_type: str = Field(default="rfdetr", title="Detector Type")
                tracker_type: str = Field(default="deepsort", title="Tracker Type")
                merger_type: str = Field(default="bev_cluster", title="Merger Type")

                @validator("detector_type")
                def validate_detector_type(self, v: str) -> str:
                    return v

                @validator("tracker_type")
                def validate_tracker_type(self, v: str) -> str:
                    return v

                @validator("merger_type")
                def validate_merger_type(self, v: str) -> str:
                    return v

                def get_detector_config(self) -> BaseDetectorConfig:
                    if self.detector_type == "rfdetr":
                        return RFDETRDetectorConfig()
                    if self.detector_type == "yolo":
                        return YOLODetectorConfig()
                    if self.detector_type == "dummy":
                        return DummyDetectorConfig()
                    raise ValueError(f"Unknown detector type: {self.detector_type}")

                def get_tracker_config(self) -> BaseTrackerConfig:
                    if self.tracker_type == "deepsort":
                        return DeepSORTTrackerConfig()
                    if self.tracker_type == "bytetrack":
                        return ByteTrackTrackerConfig()
                    if self.tracker_type == "botsort":
                        return BoTSORTTrackerConfig()
                    if self.tracker_type == "dummy":
                        return DummyTrackerConfig()
                    raise ValueError(f"Unknown tracker type: {self.tracker_type}")

                def get_merger_config(self) -> BaseModel:
                    if self.merger_type == "bev_cluster":
                        return CrossCameraConfig()
                    raise ValueError(f"Unknown merger type: {self.merger_type}")

                def get_available_detectors(self) -> list[str]:
                    return ["rfdetr", "yolo", "dummy"]

                def get_available_trackers(self) -> list[str]:
                    return ["deepsort", "bytetrack", "botsort", "dummy"]

                def get_available_mergers(self) -> list[str]:
                    return ["bev_cluster"]

            _VisionSystemConfig = _VisionSystemConfigFallback

    return _VisionSystemConfig


def refresh_config_system() -> None:
    global _VisionSystemConfig, _DetectorType, _TrackerType, _MergerType  # noqa: PLW0603
    _VisionSystemConfig = None
    _DetectorType = None
    _TrackerType = None
    _MergerType = None


class VisionSystemConfigMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return get_vision_system_config(force_refresh=False)(*args, **kwargs)

    def model_json_schema(cls) -> dict[str, Any]:
        return get_vision_system_config(force_refresh=False).model_json_schema()

    def model_dump_json(cls, *args: Any, **kwargs: Any) -> str:
        return get_vision_system_config(force_refresh=False).model_dump_json(*args, **kwargs)


class VisionSystemConfig(metaclass=VisionSystemConfigMeta):
    """Dynamic Vision System Configuration proxy."""

    pass


def get_detector_type() -> str:
    if _DetectorType is None:
        get_vision_system_config()
    return _DetectorType or "str"


def get_tracker_type() -> str:
    if _TrackerType is None:
        get_vision_system_config()
    return _TrackerType or "str"


DetectorType = Literal["rfdetr", "yolo", "dummy"]
TrackerType = Literal["deepsort", "bytetrack", "dummy"]
MergerType = Literal["bev_cluster"]
