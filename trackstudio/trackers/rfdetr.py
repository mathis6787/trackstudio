"""Compatibility module for the old RF-DETR tracker name."""

from .deepsort import DeepSORTSingleCameraTracker

RFDETRTracker = DeepSORTSingleCameraTracker

__all__ = ["RFDETRTracker", "DeepSORTSingleCameraTracker"]
