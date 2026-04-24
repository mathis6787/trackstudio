"""Configuration registry for detector, tracker, and merger components."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel, Field, create_model

from trackstudio.detectors.base import BaseDetectorConfig
from trackstudio.trackers.base import BaseTrackerConfig

logger = logging.getLogger(__name__)

_DETECTOR_CONFIGS: dict[str, type[BaseDetectorConfig]] = {}
_TRACKER_CONFIGS: dict[str, type[BaseTrackerConfig]] = {}
_MERGER_CONFIGS: dict[str, type[BaseModel]] = {}


def _refresh_config_system() -> None:
    try:
        from .vision_config import refresh_config_system  # noqa: PLC0415

        refresh_config_system()
    except ImportError:
        pass


def register_detector_config(name: str) -> Callable[[type[BaseDetectorConfig]], type[BaseDetectorConfig]]:
    """Register a detector configuration class."""

    def decorator(config_class: type[BaseDetectorConfig]) -> type[BaseDetectorConfig]:
        if not issubclass(config_class, BaseDetectorConfig):
            raise ValueError(f"Config class {config_class.__name__} must inherit from BaseDetectorConfig")

        _DETECTOR_CONFIGS[name] = config_class
        logger.debug(f"📝 Registered detector config: {name} -> {config_class.__name__}")
        _refresh_config_system()
        return config_class

    return decorator


def register_tracker_config(name: str) -> Callable[[type[BaseTrackerConfig]], type[BaseTrackerConfig]]:
    """Register a tracker configuration class."""

    def decorator(config_class: type[BaseTrackerConfig]) -> type[BaseTrackerConfig]:
        if not issubclass(config_class, BaseTrackerConfig):
            raise ValueError(f"Config class {config_class.__name__} must inherit from BaseTrackerConfig")

        _TRACKER_CONFIGS[name] = config_class
        logger.debug(f"📝 Registered tracker config: {name} -> {config_class.__name__}")
        _refresh_config_system()
        return config_class

    return decorator


def register_merger_config(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Register a merger configuration class."""

    def decorator(config_class: type[BaseModel]) -> type[BaseModel]:
        if not issubclass(config_class, BaseModel):
            raise ValueError(f"Config class {config_class.__name__} must inherit from BaseModel")

        _MERGER_CONFIGS[name] = config_class
        logger.debug(f"📝 Registered merger config: {name} -> {config_class.__name__}")
        _refresh_config_system()
        return config_class

    return decorator


def get_registered_detector_configs() -> dict[str, type[BaseDetectorConfig]]:
    return _DETECTOR_CONFIGS.copy()


def get_registered_tracker_configs() -> dict[str, type[BaseTrackerConfig]]:
    return _TRACKER_CONFIGS.copy()


def get_registered_merger_configs() -> dict[str, type[BaseModel]]:
    return _MERGER_CONFIGS.copy()


def get_detector_names() -> list[str]:
    return list(_DETECTOR_CONFIGS.keys())


def get_tracker_names() -> list[str]:
    return list(_TRACKER_CONFIGS.keys())


def get_merger_names() -> list[str]:
    return list(_MERGER_CONFIGS.keys())


def create_vision_system_config() -> tuple[type[BaseModel], str, str, str]:
    """Create the dynamic system config model from registered component configs."""
    detector_names = get_detector_names()
    tracker_names = get_tracker_names()
    merger_names = get_merger_names() or ["bev_cluster"]

    if not detector_names:
        raise ValueError("No detector configurations registered")
    if not tracker_names:
        raise ValueError("No tracker configurations registered")

    detector_type_union = "|".join(detector_names)
    tracker_type_union = "|".join(tracker_names)
    merger_type_union = "|".join(merger_names)

    field_definitions = {
        "detector_type": (
            str,
            Field(
                default="rfdetr" if "rfdetr" in detector_names else detector_names[0],
                title="Detector Type",
                description=f"Object detector to use. Available: {', '.join(detector_names)}",
            ),
        ),
        "tracker_type": (
            str,
            Field(
                default="deepsort" if "deepsort" in tracker_names else tracker_names[0],
                title="Tracker Type",
                description=f"Single-camera tracker to use. Available: {', '.join(tracker_names)}",
            ),
        ),
        "merger_type": (
            str,
            Field(
                default="bev_cluster" if "bev_cluster" in merger_names else merger_names[0],
                title="Merger Type",
                description=f"Cross-camera merger to use. Available: {', '.join(merger_names)}",
            ),
        ),
    }

    for detector_name in detector_names:
        config_class = _DETECTOR_CONFIGS.get(detector_name)
        if config_class:
            field_definitions[f"{detector_name}_detector"] = (
                config_class,
                Field(default_factory=config_class, title=f"{detector_name.title()} Detector Config"),
            )

    for tracker_name in tracker_names:
        config_class = _TRACKER_CONFIGS.get(tracker_name)
        if config_class:
            field_definitions[f"{tracker_name}_tracker"] = (
                config_class,
                Field(default_factory=config_class, title=f"{tracker_name.title()} Tracker Config"),
            )

    for merger_name in merger_names:
        config_class = _MERGER_CONFIGS.get(merger_name)
        if config_class:
            field_definitions[f"{merger_name}_merger"] = (
                config_class,
                Field(default_factory=config_class, title=f"{merger_name.title()} Merger Config"),
            )

    vision_system_config = create_model("VisionSystemConfig", **field_definitions, __base__=BaseModel)
    vision_system_config.model_rebuild()

    def get_detector_config(self: BaseModel) -> BaseDetectorConfig:
        detector_type = getattr(self, "detector_type", None)
        field_name = f"{detector_type}_detector"
        if hasattr(self, field_name):
            return cast(BaseDetectorConfig, getattr(self, field_name))
        raise ValueError(f"Unknown detector type: {detector_type}")

    def get_tracker_config(self: BaseModel) -> BaseTrackerConfig:
        tracker_type = getattr(self, "tracker_type", None)
        field_name = f"{tracker_type}_tracker"
        if hasattr(self, field_name):
            return cast(BaseTrackerConfig, getattr(self, field_name))
        raise ValueError(f"Unknown tracker type: {tracker_type}")

    def get_merger_config(self: BaseModel) -> BaseModel:
        merger_type = getattr(self, "merger_type", None)
        field_name = f"{merger_type}_merger"
        if hasattr(self, field_name):
            return cast(BaseModel, getattr(self, field_name))
        raise ValueError(f"Unknown merger type: {merger_type}")

    def get_available_detectors(_self: BaseModel) -> list[str]:
        return get_detector_names()

    def get_available_trackers(_self: BaseModel) -> list[str]:
        return get_tracker_names()

    def get_available_mergers(_self: BaseModel) -> list[str]:
        return get_merger_names()

    vision_system_config.get_detector_config = get_detector_config  # type: ignore[attr-defined]
    vision_system_config.get_tracker_config = get_tracker_config  # type: ignore[attr-defined]
    vision_system_config.get_merger_config = get_merger_config  # type: ignore[attr-defined]
    vision_system_config.get_available_detectors = get_available_detectors  # type: ignore[attr-defined]
    vision_system_config.get_available_trackers = get_available_trackers  # type: ignore[attr-defined]
    vision_system_config.get_available_mergers = get_available_mergers  # type: ignore[attr-defined]

    return vision_system_config, detector_type_union, tracker_type_union, merger_type_union


def import_all_configs() -> None:
    """Import config modules to trigger registration."""
    try:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("trackstudio.vision_config") is not None:
            logger.debug("📦 Vision config module available")
    except ImportError as e:
        logger.warning(f"⚠️ Could not import vision configs: {e}")

    logger.info(
        f"📋 Config registry loaded: {len(_DETECTOR_CONFIGS)} detectors, "
        f"{len(_TRACKER_CONFIGS)} trackers, {len(_MERGER_CONFIGS)} mergers"
    )


def get_config_classes() -> tuple[type[BaseModel], str, str, str]:
    import_all_configs()
    return create_vision_system_config()
