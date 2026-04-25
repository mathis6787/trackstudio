"""Detector interfaces."""

from .base import BaseDetectorConfig, VisionDetector

__all__ = ["BaseDetectorConfig", "VisionDetector", "DummyDetector", "RFDETRDetector", "YOLODetector"]


def __getattr__(name: str):
    if name == "DummyDetector":
        from .dummy import DummyDetector  # noqa: PLC0415

        return DummyDetector
    if name == "RFDETRDetector":
        from .rfdetr import RFDETRDetector  # noqa: PLC0415

        return RFDETRDetector
    if name == "YOLODetector":
        from .yolo import YOLODetector  # noqa: PLC0415

        return YOLODetector
    raise AttributeError(name)
