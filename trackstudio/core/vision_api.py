"""
Vision API Wrapper - Modular Computer Vision Processing

This module provides a clean API interface for integrating computer vision
processing into the WebRTC video stream. It's designed to be easily extensible
with different tracking algorithms through submodules.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ..bev import BEVTransformer
from ..detectors import VisionDetector
from ..mergers import VisionMerger
from ..trackers import SingleCameraTracker
from ..vision_config import VisionSystemConfig
from ..vision_factory import create_vision_system  # noqa: PLC0415
from ..vision_types import VisionResult

logger = logging.getLogger(__name__)


class VisionAPI:
    """Main Vision API - coordinates all vision processing"""

    def __init__(
        self,
        config: VisionSystemConfig | None = None,
        detector: VisionDetector | None = None,
        tracker: SingleCameraTracker | None = None,
        bev_transformer: BEVTransformer | None = None,
        merger: VisionMerger | None = None,
    ):
        if detector is None or tracker is None or bev_transformer is None or merger is None:
            detector_type = getattr(config, "detector_type", None) if config else None
            tracker_type = getattr(config, "tracker_type", None) if config else None
            merger_type = getattr(config, "merger_type", None) if config else None
            components = create_vision_system(detector_type, tracker_type, merger_type)

            self.detector = detector or components.detector
            self.tracker = tracker or components.tracker
            self.bev_transformer = bev_transformer or components.bev_transformer
            self.merger = merger or components.merger
            self.config = config or components.config
        else:
            self.detector = detector
            self.tracker = tracker
            self.bev_transformer = bev_transformer
            self.merger = merger
            self.config = config or VisionSystemConfig()

        self.frame_counter = 0
        self.is_enabled = False
        self.processing_times = []
        self.max_history = 100

        # ⚡ VISION FPS CONTROL - Process vision at 10fps while UI runs at 30fps
        self.vision_fps = 10.0  # Configurable vision processing FPS
        self.vision_frame_interval = 1.0 / self.vision_fps  # 0.1 seconds between vision frames
        self.last_vision_time = 0.0
        self.cached_vision_result: VisionResult | None = None
        self.vision_frame_counter = 0  # Separate counter for vision frames

        detector_name = self.detector.__class__.__name__
        tracker_name = self.tracker.__class__.__name__
        merger_name = self.merger.__class__.__name__
        logger.info(f"🧠 VisionAPI initialized with {detector_name}, {tracker_name} and {merger_name}")
        logger.info(f"🎯 Vision processing FPS: {self.vision_fps} (UI stream can run at full speed)")

    def enable_tracking(self):
        """Enable vision processing"""
        self.is_enabled = True
        logger.info("✅ Vision tracking enabled")

    def disable_tracking(self):
        """Disable vision processing"""
        self.is_enabled = False
        logger.info("❌ Vision tracking disabled")

    def is_tracking_enabled(self) -> bool:
        """Check if tracking is enabled"""
        return self.is_enabled

    def set_vision_fps(self, fps: float):
        """Set the vision processing FPS (independent of UI stream FPS)"""
        self.vision_fps = max(1.0, min(30.0, fps))  # Clamp between 1-30 FPS
        self.vision_frame_interval = 1.0 / self.vision_fps
        logger.info(f"🎯 Vision processing FPS changed to: {self.vision_fps}")

    def get_vision_fps(self) -> float:
        """Get the current vision processing FPS"""
        return self.vision_fps

    def process_combined_frame(
        self, combined_frame: np.ndarray, timestamp: float, num_streams: int = 2, stream_ids: list[int] | None = None
    ) -> VisionResult | None:
        """
        Process a combined frame with dynamic number of streams and return vision results

        Args:
            combined_frame: Combined frame from multiple streams
            timestamp: Frame timestamp
            num_streams: Number of active streams (1-4)
            stream_ids: List of active stream IDs

        Returns:
            VisionResult with detections, tracks, and BEV coordinates for all streams
        """
        if not self.is_enabled:
            logger.debug(f"🚫 Vision processing disabled - frame {self.frame_counter + 1} skipped")
            return None

        self.frame_counter += 1

        # ⚡ VISION FPS CONTROL - Only process vision at specified FPS
        current_time = time.time()
        time_since_last_vision = current_time - self.last_vision_time

        if time_since_last_vision < self.vision_frame_interval and self.cached_vision_result:
            # Update frame counter and timestamp for smooth UI
            cached_result = VisionResult(
                frame_id=self.frame_counter,
                timestamp=timestamp,
                bev_tracks=self.cached_vision_result.bev_tracks,
                processing_time_ms=0.0,  # No processing time for cached result
                num_streams=num_streams,
                active_stream_ids=stream_ids or list(range(num_streams)),
                all_stream_detections=self.cached_vision_result.all_stream_detections,
                all_stream_tracks=self.cached_vision_result.all_stream_tracks,
            )
            logger.debug(f"⚡ Vision frame {self.frame_counter} SKIPPED - using cached result (fps={self.vision_fps})")
            return cached_result

        # Process this frame - time for new vision processing
        self.last_vision_time = current_time
        self.vision_frame_counter += 1

        overall_start_time = time.time()

        try:
            if stream_ids is None:
                stream_ids = list(range(num_streams))

            # ⏱️ STEP 1: Frame Layout Calculation
            step1_start = time.time()
            # Calculate layout based on number of streams
            if num_streams == 1:
                grid_cols, _grid_rows = 1, 1
            elif num_streams == 2:
                grid_cols, _grid_rows = 2, 1
            elif num_streams in {3, 4}:
                grid_cols, _grid_rows = 2, 2
            else:
                grid_cols, _grid_rows = 2, 2  # Default
            step1_time = (time.time() - step1_start) * 1000

            # ⏱️ STEP 2: Frame Extraction
            step2_start = time.time()
            # Extract individual stream frames from combined frame
            stream_frames = {}
            for i, stream_id in enumerate(stream_ids[:num_streams]):  # Limit to actual number of streams
                # Calculate grid position
                col = i % grid_cols
                row = i // grid_cols

                # Calculate position in combined frame (each stream is 720x480)
                x_start = col * 720
                y_start = row * 480
                x_end = x_start + 720
                y_end = y_start + 480

                # Extract stream frame
                h, w = combined_frame.shape[:2]
                if y_end <= h and x_end <= w:
                    stream_frame = combined_frame[y_start:y_end, x_start:x_end]
                    stream_frames[stream_id] = stream_frame

                else:
                    logger.warning(
                        f"Stream {stream_id} position ({x_start},{y_start})-({x_end},{y_end}) exceeds frame bounds ({w}x{h})"
                    )
            step2_time = (time.time() - step2_start) * 1000

            # ⏱️ STEP 3: Detection and Tracking per Stream
            step3_start = time.time()
            # Process each stream separately for detection
            all_detections = {}
            all_tracks = {}

            detection_times = {}
            tracking_times = {}

            for stream_id, frame in stream_frames.items():
                # ⏱️ Detection timing
                detection_start = time.time()
                detections = self.detector.detect(frame, camera_id=stream_id)
                detection_times[stream_id] = (time.time() - detection_start) * 1000
                all_detections[stream_id] = detections

                # ⏱️ Tracking timing
                tracking_start = time.time()
                tracks = self.tracker.track(detections, camera_id=stream_id, timestamp=timestamp, frame=frame)
                tracking_times[stream_id] = (time.time() - tracking_start) * 1000
                all_tracks[stream_id] = tracks

            step3_time = (time.time() - step3_start) * 1000

            # ⏱️ STEP 4: BEV Transformation
            step4_start = time.time()
            # Combine all tracks for BEV transformation
            combined_tracks = []
            for stream_id, tracks in all_tracks.items():
                combined_tracks.extend(tracks)
                if tracks:
                    logger.debug(f"📍 Adding {len(tracks)} tracks from stream {stream_id} to BEV transformation")

            logger.debug(f"🗺️ Total tracks for BEV transform: {len(combined_tracks)}")
            bev_tracks = self.bev_transformer.transform_to_bev(combined_tracks)
            logger.debug(f"✅ BEV transform result: {len(bev_tracks)} BEV tracks")
            step4_time = (time.time() - step4_start) * 1000

            # ⏱️ STEP 5: ReID Feature Extraction
            step5_start = time.time()
            # Extract ReID features if tracker supports it
            reid_features = {}
            if hasattr(self.tracker, "get_reid_features"):
                # Extract features for each active stream
                for stream_id, stream_frame in stream_frames.items():
                    if stream_id in all_tracks:
                        stream_tracks = all_tracks[stream_id]

                        features = self.tracker.get_reid_features(stream_frame, stream_tracks)

                        # Store features by track ID
                        if features is not None:
                            for track, feature in zip(stream_tracks, features, strict=False):
                                reid_features[track.track_id] = feature
            step5_time = (time.time() - step5_start) * 1000

            # ⏱️ STEP 6: Cross-Camera Merging
            step6_start = time.time()
            # Apply cross-camera track merging
            logger.debug(
                f"🔗 Cross-camera merge input: {len(bev_tracks)} BEV tracks, {len(reid_features)} ReID features"
            )
            unified_bev_tracks = self.merger.merge(bev_tracks, timestamp, stream_frames, reid_features)
            logger.debug(f"🌐 Cross-camera merge result: {len(unified_bev_tracks)} unified tracks")
            step6_time = (time.time() - step6_start) * 1000

            # Calculate total processing time
            total_processing_time = (time.time() - overall_start_time) * 1000

            # 📊 Log timing stats only every 30 vision frames to reduce noise
            total_detections = sum(len(dets) for dets in all_detections.values())
            total_tracks = sum(len(tracks) for tracks in all_tracks.values())

            if self.vision_frame_counter % 30 == 0:  # Only log every 30 vision frames
                print(
                    f"\n🎯 VISION PROCESSING TIMING - Vision Frame {self.vision_frame_counter} (UI Frame {self.frame_counter}) @ {self.vision_fps}fps"
                )
                print(f"   📐 Step 1 - Layout Calc:     {step1_time:.2f}ms")
                print(f"   ✂️  Step 2 - Frame Extract:   {step2_time:.2f}ms")
                print(f"   🔍 Step 3 - Detect+Track:    {step3_time:.2f}ms")
                for stream_id in stream_frames:
                    det_time = detection_times.get(stream_id, 0)
                    track_time = tracking_times.get(stream_id, 0)
                    det_count = len(all_detections.get(stream_id, []))
                    track_count = len(all_tracks.get(stream_id, []))
                    print(
                        f"      📹 Stream {stream_id}: detect={det_time:.2f}ms ({det_count} objs), track={track_time:.2f}ms ({track_count} tracks)"
                    )
                print(
                    f"   🗺️  Step 4 - BEV Transform:  {step4_time:.2f}ms ({len(combined_tracks)} → {len(bev_tracks)} tracks)"
                )
                print(f"   🆔 Step 5 - ReID Features:   {step5_time:.2f}ms ({len(reid_features)} features)")
                print(
                    f"   🔗 Step 6 - Cross-Cam Merge: {step6_time:.2f}ms ({len(bev_tracks)} → {len(unified_bev_tracks)} unified)"
                )
                print(f"   ⚡ TOTAL PROCESSING TIME:    {total_processing_time:.2f}ms")
                print(
                    f"   📊 SUMMARY: {num_streams} streams, {total_detections} detections, {total_tracks} tracks, {len(unified_bev_tracks)} final"
                )

            # Store performance metrics
            self.processing_times.append(total_processing_time)
            if len(self.processing_times) > self.max_history:
                self.processing_times.pop(0)

            # Create new vision result
            result = VisionResult(
                frame_id=self.frame_counter,
                timestamp=timestamp,
                bev_tracks=unified_bev_tracks,
                processing_time_ms=total_processing_time,
                num_streams=num_streams,
                active_stream_ids=stream_ids,
                all_stream_detections=all_detections,
                all_stream_tracks=all_tracks,
            )
            # Cache this result for skipped frames
            self.cached_vision_result = result

            # Multi-stream data is now stored directly in the VisionResult object

            if self.vision_frame_counter % 10 == 0:  # Log every 10 vision frames
                avg_time = sum(self.processing_times) / len(self.processing_times)
                logger.info(
                    f"🎯 Vision processing: vision frame {self.vision_frame_counter}, UI frame {self.frame_counter}, "
                    f"streams: {num_streams}, avg time: {avg_time:.1f}ms, "
                    f"total detections: {total_detections}, BEV tracks: {len(bev_tracks)}"
                )

            return result

        except Exception as e:
            logger.error(f"❌ Vision processing error: {e}")
            return None

    def get_statistics(self) -> dict[str, Any]:
        """Get vision processing statistics"""
        merger_stats = self.merger.get_statistics() if self.merger else {}

        if not self.processing_times:
            return {
                "enabled": self.is_enabled,
                "frames_processed": self.frame_counter,
                "vision_frames_processed": self.vision_frame_counter,
                "vision_fps": self.vision_fps,
                "vision_processing_ratio": 0.0,
                "avg_processing_time_ms": 0.0,
                "max_processing_time_ms": 0.0,
                "min_processing_time_ms": 0.0,
                "detector_stats": self.detector.get_statistics() if self.detector else {},
                "cross_camera_merging": merger_stats,
                "tracker_stats": self.tracker.get_statistics() if self.tracker else {},
            }

        # Calculate vision processing ratio (vision frames / total frames)
        processing_ratio = (self.vision_frame_counter / max(1, self.frame_counter)) * 100.0

        return {
            "enabled": self.is_enabled,
            "frames_processed": self.frame_counter,
            "vision_frames_processed": self.vision_frame_counter,
            "vision_fps": self.vision_fps,
            "vision_processing_ratio": round(processing_ratio, 1),
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times),
            "max_processing_time_ms": max(self.processing_times),
            "min_processing_time_ms": min(self.processing_times),
            "detector_stats": self.detector.get_statistics() if self.detector else {},
            "cross_camera_merging": merger_stats,
            "tracker_stats": self.tracker.get_statistics() if self.tracker else {},
        }

    def set_detector(self, detector: VisionDetector):
        """Set a new detector."""
        self.detector = detector
        logger.info(f"🔄 Vision detector changed to: {detector.__class__.__name__}")

    def set_tracker(self, tracker: SingleCameraTracker):
        """Set a new vision tracker (for swapping algorithms)"""
        self.tracker = tracker
        logger.info(f"🔄 Vision tracker changed to: {tracker.__class__.__name__}")

    def update_homography(self, camera_id: int, homography_matrix: np.ndarray):
        """Update homography matrix for a specific camera"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            calibration.update_homography(camera_id, homography_matrix)
        else:
            logger.warning("BEV transformer does not support calibration")

    # Calibration delegation methods
    def calibrate_camera(
        self,
        camera_id: int,
        image_points: list[tuple[float, float]],
        bev_points: list[tuple[float, float]],
        bev_size: int = 400,
    ) -> tuple[bool, str, np.ndarray | None]:
        """Delegate calibration to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            return calibration.calibrate_camera(camera_id, image_points, bev_points, bev_size)
        return False, "BEV transformer does not support calibration", None

    def transform_image_with_homography(
        self, image: np.ndarray, camera_id: int, output_size: tuple[int, int] = (400, 400)
    ) -> np.ndarray | None:
        """Delegate image transformation to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            return calibration.transform_image_with_homography(image, camera_id, output_size)
        return None

    def save_calibration_data(
        self,
        camera_id: int,
        image_points: list[tuple[float, float]],
        bev_points: list[tuple[float, float]],
        homography_matrix: np.ndarray,
        bev_size: int = 400,
    ):
        """Delegate calibration data saving to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            calibration.save_calibration_data(camera_id, image_points, bev_points, homography_matrix, bev_size)
        else:
            logger.warning("BEV transformer does not support calibration data saving")

    def load_calibration_data(self):
        """Delegate calibration data loading to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            return calibration.load_calibration_data()
        return {}

    def clear_calibration_data(self):
        """Delegate calibration data clearing to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            calibration.clear_calibration_data()
        else:
            logger.warning("BEV transformer does not support calibration data clearing")

    def get_calibration_status(self) -> dict[str, Any]:
        """Delegate calibration status retrieval to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            return calibration.get_calibration_status()
        return {
            "camera0": {"calibrated": False},
            "camera1": {"calibrated": False},
            "camera2": {"calibrated": False},
            "camera3": {"calibrated": False},
        }

    def get_config_schema(self) -> dict[str, Any] | None:
        """Get the JSON schema for the current vision system's configuration"""
        if self.config:
            schema = self.config.model_json_schema()

            # Filter schema to only show currently active detector/tracker/merger configs
            if "properties" in schema:
                current_detector = self.config.detector_type
                current_tracker = self.config.tracker_type
                current_merger = self.config.merger_type

                # Log what we're filtering
                all_props = list(schema["properties"].keys())
                logger.info(
                    f"📋 Config schema filtering: active detector='{current_detector}', "
                    f"tracker='{current_tracker}', merger='{current_merger}'"
                )
                logger.debug(f"   All properties: {all_props}")

                # Keep only relevant properties
                filtered_properties = {}

                # Keep only the currently active detector config
                active_detector_key = f"{current_detector}_detector"
                if active_detector_key in schema["properties"]:
                    filtered_properties[active_detector_key] = schema["properties"][active_detector_key]

                # Keep only the currently active tracker config
                active_tracker_key = f"{current_tracker}_tracker"
                if active_tracker_key in schema["properties"]:
                    filtered_properties[active_tracker_key] = schema["properties"][active_tracker_key]

                # Keep only the currently active merger config
                active_merger_key = f"{current_merger}_merger"
                if active_merger_key in schema["properties"]:
                    filtered_properties[active_merger_key] = schema["properties"][active_merger_key]

                # Update schema with filtered properties
                schema["properties"] = filtered_properties

                # Log what we kept
                kept_props = list(filtered_properties.keys())
                logger.info(f"   Filtered properties: {kept_props} (removed {len(all_props) - len(kept_props)} fields)")

                # Also update required fields if present
                if "required" in schema:
                    schema["required"] = [field for field in schema.get("required", []) if field in filtered_properties]

            # Resolve refs for frontend
            if "$defs" in schema:
                defs = schema["$defs"]

                def resolve_refs(obj):
                    """Recursively resolve $ref references in the schema"""
                    if isinstance(obj, dict):
                        if "$ref" in obj:
                            ref_key = obj["$ref"].split("/")[-1]
                            if ref_key in defs:
                                resolved = defs[ref_key].copy()
                                # Preserve title from the original reference
                                if "title" in obj:
                                    resolved["title"] = obj["title"]
                                return resolve_refs(resolved)
                        else:
                            return {k: resolve_refs(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [resolve_refs(item) for item in obj]
                    return obj

                # Resolve all references in properties
                schema["properties"] = resolve_refs(schema["properties"])
                del schema["$defs"]
            return schema
        return None

    def get_current_config(self) -> VisionSystemConfig | None:
        """Get the current vision system's configuration"""
        return self.config

    def update_config(self, config_update: dict[str, Any]):
        """Update the vision system's configuration"""
        if not self.config:
            return

        def deep_merge(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
            for key, value in update.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        current_config_dict = self.config.model_dump()
        deep_merge(current_config_dict, config_update)

        self.config = VisionSystemConfig(**current_config_dict)

        # Pass the updated sub-configs to the respective components
        if self.detector:
            detector_config = self.config.get_detector_config()
            self.detector.update_config(detector_config.model_dump())

        if self.tracker:
            tracker_config = self.config.get_tracker_config()
            self.tracker.update_config(tracker_config.model_dump())

        # Update merger config
        if hasattr(self.merger, "config") and hasattr(self.config, "get_merger_config"):
            try:
                merger_config = self.config.get_merger_config()
                self.merger.config = merger_config
            except Exception as e:
                logger.warning(f"⚠️ Could not update merger config: {e}")

        logger.info(f"✅ VisionAPI configuration updated: {config_update}")

    def restart_vision_system(self, preserve_calibration: bool = True):
        """
        Restart the entire vision system with fresh tracker and merger instances.
        This resets all tracking states and reinitializes components with current configuration.
        Args:
            preserve_calibration: Whether to preserve existing calibration data (default: True)
        """
        logger.info("🔄 Restarting vision system...")

        # Store current state
        was_enabled = self.is_enabled
        current_vision_fps = self.vision_fps

        # Preserve calibration data if requested
        calibration_data = None
        if preserve_calibration:
            try:
                calibration_data = self.load_calibration_data()
                logger.info(f"💾 Preserved calibration data for {len(calibration_data)} cameras")
            except Exception as e:
                logger.warning(f"⚠️ Could not preserve calibration data: {e}")

        # Get calibration file path from current BEV transformer
        calibration_file = None
        if hasattr(self.bev_transformer, "calibration") and hasattr(self.bev_transformer.calibration, "calibration_file"):
            calibration_file = self.bev_transformer.calibration.calibration_file
            logger.info(f"💾 Preserved calibration file path: {calibration_file}")

        # Temporarily disable tracking
        self.disable_tracking()

        # Reset all counters and state
        self.frame_counter = 0
        self.vision_frame_counter = 0
        self.last_vision_time = 0.0
        self.cached_vision_result = None
        self.processing_times.clear()

        # Recreate detector, tracker, BEV transformer and merger with current configuration
        try:
            from ..vision_factory import create_vision_system  # noqa: PLC0415

            detector_type = getattr(self.config, "detector_type", "rfdetr")
            tracker_type = getattr(self.config, "tracker_type", "deepsort")
            merger_type = getattr(self.config, "merger_type", "bev_cluster")

            logger.info(f"🏭 Creating new detector: {detector_type}, tracker: {tracker_type}, merger: {merger_type}")
            components = create_vision_system(detector_type, tracker_type, merger_type, calibration_file)

            # Replace old components
            old_detector_name = self.detector.__class__.__name__
            old_tracker_name = self.tracker.__class__.__name__
            old_merger_name = self.merger.__class__.__name__

            self.detector = components.detector
            self.tracker = components.tracker
            self.bev_transformer = components.bev_transformer
            self.merger = components.merger

            logger.info(f"✅ Replaced {old_detector_name} -> {self.detector.__class__.__name__}")
            logger.info(f"✅ Replaced {old_tracker_name} -> {self.tracker.__class__.__name__}")
            logger.info(f"✅ Replaced {old_merger_name} -> {self.merger.__class__.__name__}")

            # Restore calibration data if preserved
            if preserve_calibration and calibration_data:
                try:
                    for camera_key, cam_data in calibration_data.items():
                        if camera_key.startswith("camera") and "homography_matrix" in cam_data:
                            camera_id = int(camera_key.replace("camera", ""))
                            matrix = np.array(cam_data["homography_matrix"], dtype=np.float32)
                            self.update_homography(camera_id, matrix)
                            logger.info(f"📐 Restored homography for camera {camera_id}")
                except Exception as e:
                    logger.error(f"❌ Error restoring calibration data: {e}")

            # Apply current configuration to new components
            if self.config:
                try:
                    detector_config = self.config.get_detector_config()
                    self.detector.update_config(detector_config.model_dump())
                    logger.info("⚙️ Applied current detector configuration")

                    tracker_config = self.config.get_tracker_config()
                    self.tracker.update_config(tracker_config.model_dump())
                    logger.info("⚙️ Applied current tracker configuration")

                    if hasattr(self.merger, "config") and hasattr(self.config, "get_merger_config"):
                        merger_config = self.config.get_merger_config()
                        self.merger.config = merger_config
                        logger.info("⚙️ Applied current merger configuration")
                except Exception as e:
                    logger.warning(f"⚠️ Could not apply configuration to new components: {e}")

            # Restore previous state
            self.set_vision_fps(current_vision_fps)
            if was_enabled:
                self.enable_tracking()

            logger.info("🎉 Vision system restart completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Vision system restart failed: {e}")
            # Try to restore original state
            if was_enabled:
                self.enable_tracking()
            return False

    def get_homography_matrix(self, camera_id: int) -> np.ndarray | None:
        """Delegate homography matrix retrieval to the BEV transformer's calibration module"""
        calibration = getattr(self.bev_transformer, "calibration", None)
        if calibration:
            return calibration.get_homography_matrix(camera_id)
        return None


def create_vision_api(
    detector_type: str | None = None,
    tracker_type: str | None = None,
    merger_type: str | None = None,
    calibration_file: str | None = None,
) -> VisionAPI:
    """
    Create a new VisionAPI instance with specified detector, tracker and merger types.

    Args:
        detector_type: Type of detector to use ("rfdetr", "yolo", "dummy", or None for default)
        tracker_type: Type of tracker to use ("deepsort", "bytetrack", "dummy", or None for default)
        merger_type: Type of merger to use ("bev_cluster", or None for default)
        calibration_file: Optional path to calibration data file

    Returns:
        VisionAPI instance configured with the specified tracker and merger
    """
    from trackstudio.vision_factory import create_vision_system  # noqa: PLC0415

    logger.info(
        f"🆕 Creating VisionAPI with detector: {detector_type or 'default'}, tracker: {tracker_type or 'default'}"
    )
    components = create_vision_system(detector_type, tracker_type, merger_type, calibration_file)
    api = VisionAPI(
        config=components.config,
        detector=components.detector,
        tracker=components.tracker,
        bev_transformer=components.bev_transformer,
        merger=components.merger,
    )

    logger.info(
        f"✅ VisionAPI created with detector: {api.detector.__class__.__name__}, "
        f"tracker: {api.tracker.__class__.__name__}"
    )
    return api


# Backward compatibility
def get_vision_api(
    detector_type: str | None = None,
    tracker_type: str | None = None,
    merger_type: str | None = None,
    calibration_file: str | None = None,
) -> VisionAPI:
    """Create a new VisionAPI instance (backward compatibility)"""
    return create_vision_api(detector_type, tracker_type, merger_type, calibration_file)
