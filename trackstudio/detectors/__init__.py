"""Detector interfaces."""

from .base import BaseDetectorConfig, VisionDetector

__all__ = ["BaseDetectorConfig", "VisionDetector", "DummyDetector", "RFDETRDetector"]


def __getattr__(name: str):
    if name == "DummyDetector":
        from .dummy import DummyDetector  # noqa: PLC0415

        return DummyDetector
    if name == "RFDETRDetector":
        from .rfdetr import RFDETRDetector  # noqa: PLC0415

        return RFDETRDetector
    raise AttributeError(name)
