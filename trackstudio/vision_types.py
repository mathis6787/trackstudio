"""Shared vision data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Single object detection result."""

    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int


@dataclass
class Track:
    """Single-camera tracked object."""

    track_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    age: int
    camera_id: int


@dataclass
class BEVTrack:
    """Tracked object transformed into bird's-eye-view coordinates."""

    track_id: str
    bev_x: float
    bev_y: float
    confidence: float
    camera_id: int
    global_id: int | None = None
    trajectory: list[tuple[float, float, float]] | None = None


@dataclass
class VisionResult:
    """Complete vision processing result for all active streams."""

    frame_id: int
    timestamp: float
    bev_tracks: list[BEVTrack]
    processing_time_ms: float
    num_streams: int
    active_stream_ids: list[int]
    all_stream_detections: dict[int, list[Detection]]
    all_stream_tracks: dict[int, list[Track]]
