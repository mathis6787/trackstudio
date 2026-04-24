"""Bird's-eye-view transformation for tracked objects."""

from __future__ import annotations

import logging

from trackstudio.calibration import CameraCalibration
from trackstudio.vision_types import BEVTrack, Track

logger = logging.getLogger(__name__)


class BEVTransformer:
    """Transforms camera-space tracks into BEV coordinates."""

    def __init__(self, calibration_file: str | None = None) -> None:
        self.calibration = CameraCalibration(calibration_file) if calibration_file else CameraCalibration()

    def transform_to_bev(self, tracks: list[Track]) -> list[BEVTrack]:
        bev_tracks: list[BEVTrack] = []

        camera_track_counts: dict[int, int] = {}
        for track in tracks:
            camera_track_counts[track.camera_id] = camera_track_counts.get(track.camera_id, 0) + 1
        if camera_track_counts:
            logger.info(f"🗺️ BEV Transform input: {camera_track_counts} tracks per camera")

        for track in tracks:
            x, y, w, h = track.bbox
            feet_x = x + w // 2
            feet_y = y + h

            transformed = self.calibration.transform_points_to_bev([(feet_x, feet_y)], track.camera_id)
            if not transformed:
                logger.warning(f"❌ Camera {track.camera_id} track {track.track_id}: no homography available")
                continue

            bev_x, bev_y = transformed[0]
            if not (0 <= bev_x <= 600 and 0 <= bev_y <= 600):
                logger.debug(f"🗺️ Track {track.track_id} BEV out-of-bounds ({bev_x:.0f}, {bev_y:.0f})")

            bev_tracks.append(
                BEVTrack(
                    track_id=track.track_id,
                    bev_x=float(bev_x),
                    bev_y=float(bev_y),
                    confidence=track.confidence,
                    camera_id=track.camera_id,
                )
            )

        return bev_tracks
