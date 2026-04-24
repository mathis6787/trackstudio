from __future__ import annotations

import os
import sys
import types

import numpy as np

from trackstudio.core.vision_api import VisionAPI
from trackstudio.detectors.base import BaseDetectorConfig, VisionDetector
from trackstudio.torch_runtime import configure_torch_runtime
from trackstudio.trackers.base import BaseTrackerConfig, SingleCameraTracker
from trackstudio.vision_config import VisionSystemConfig
from trackstudio.vision_factory import create_vision_system
from trackstudio.vision_types import BEVTrack, Detection, Track


def test_torch_runtime_enables_mps_fallback(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    configure_torch_runtime()

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_vision_system_config_exposes_split_component_configs():
    config = VisionSystemConfig(detector_type="rfdetr", tracker_type="bytetrack", merger_type="bev_cluster")
    botsort_config = VisionSystemConfig(detector_type="rfdetr", tracker_type="botsort", merger_type="bev_cluster")

    assert config.detector_type == "rfdetr"
    assert config.tracker_type == "bytetrack"
    assert config.merger_type == "bev_cluster"
    assert type(config.get_detector_config()).__name__ == "RFDETRDetectorConfig"
    assert type(config.get_tracker_config()).__name__ == "ByteTrackTrackerConfig"
    assert type(botsort_config.get_tracker_config()).__name__ == "BoTSORTTrackerConfig"


def test_create_vision_system_wires_dummy_components():
    components = create_vision_system(detector_type="dummy", tracker_type="dummy", merger_type="bev_cluster")

    assert type(components.detector).__name__ == "DummyDetector"
    assert type(components.tracker).__name__ == "DummySingleCameraTracker"
    assert type(components.bev_transformer).__name__ == "BEVTransformer"
    assert type(components.merger).__name__ == "BEVClusterMerger"
    assert components.config.detector_type == "dummy"
    assert components.config.tracker_type == "dummy"


def test_create_vision_system_rejects_unknown_components():
    try:
        create_vision_system(detector_type="unknown", tracker_type="dummy", merger_type="bev_cluster")
    except ValueError as exc:
        assert "Unsupported detector type" in str(exc)
    else:
        raise AssertionError("Expected unknown detector to fail")

    try:
        create_vision_system(detector_type="dummy", tracker_type="unknown", merger_type="bev_cluster")
    except ValueError as exc:
        assert "Unsupported tracker type" in str(exc)
    else:
        raise AssertionError("Expected unknown tracker to fail")


def test_factories_create_rfdetr_trackers_with_stubbed_dependencies(monkeypatch):
    rfdetr_module = types.ModuleType("rfdetr")
    detr_module = types.ModuleType("rfdetr.detr")

    class FakeRFDETR:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, frame):
            class FakeDetections:
                class_id = []
                confidence = []
                xyxy = []

                def __len__(self):
                    return 0

            return FakeDetections()

    detr_module.RFDETRBase = FakeRFDETR
    detr_module.RFDETRLarge = FakeRFDETR
    rfdetr_module.detr = detr_module

    trackers_module = types.ModuleType("trackers")

    class FakeDeepSORTTracker:
        def __init__(self, *args, **kwargs):
            pass

    trackers_module.DeepSORTTracker = FakeDeepSORTTracker

    supervision_module = types.ModuleType("supervision")
    supervision_module.__path__ = []

    class FakeDetections:
        @classmethod
        def empty(cls):
            return cls()

        def __init__(self, *args, **kwargs):
            self.xyxy = []
            self.confidence = []
            self.class_id = []
            self.tracker_id = []

    supervision_module.Detections = FakeDetections

    supervision_tracker_module = types.ModuleType("supervision.tracker")
    supervision_tracker_module.__path__ = []
    supervision_byte_tracker_module = types.ModuleType("supervision.tracker.byte_tracker")
    supervision_byte_tracker_module.__path__ = []
    byte_core_module = types.ModuleType("supervision.tracker.byte_tracker.core")

    class FakeByteTrack:
        def __init__(self, *args, **kwargs):
            pass

    byte_core_module.ByteTrack = FakeByteTrack

    boxmot_module = types.ModuleType("boxmot")
    boxmot_module.__path__ = []
    boxmot_trackers_module = types.ModuleType("boxmot.trackers")
    boxmot_trackers_module.__path__ = []
    boxmot_botsort_package = types.ModuleType("boxmot.trackers.botsort")
    boxmot_botsort_package.__path__ = []
    boxmot_botsort_module = types.ModuleType("boxmot.trackers.botsort.botsort")

    class FakeBoTSORT:
        def __init__(self, *args, **kwargs):
            pass

    boxmot_botsort_module.BotSort = FakeBoTSORT

    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr_module)
    monkeypatch.setitem(sys.modules, "rfdetr.detr", detr_module)
    monkeypatch.setitem(sys.modules, "trackers", trackers_module)
    monkeypatch.setitem(sys.modules, "boxmot", boxmot_module)
    monkeypatch.setitem(sys.modules, "boxmot.trackers", boxmot_trackers_module)
    monkeypatch.setitem(sys.modules, "boxmot.trackers.botsort", boxmot_botsort_package)
    monkeypatch.setitem(sys.modules, "boxmot.trackers.botsort.botsort", boxmot_botsort_module)
    monkeypatch.setitem(sys.modules, "supervision", supervision_module)
    monkeypatch.setitem(sys.modules, "supervision.tracker", supervision_tracker_module)
    monkeypatch.setitem(sys.modules, "supervision.tracker.byte_tracker", supervision_byte_tracker_module)
    monkeypatch.setitem(sys.modules, "supervision.tracker.byte_tracker.core", byte_core_module)
    monkeypatch.setattr(
        "trackstudio.trackers.deepsort.DeepSORTSingleCameraTracker._initialize_reid",
        lambda self: setattr(self, "reid_extractor", object()),
    )
    monkeypatch.setattr(
        "trackstudio.trackers.botsort.BoTSORTSingleCameraTracker._initialize_reid_model",
        lambda self: setattr(self, "reid_adapter", object()),
    )

    rfdetr_deepsort = create_vision_system("rfdetr", "deepsort", "bev_cluster")
    rfdetr_bytetrack = create_vision_system("rfdetr", "bytetrack", "bev_cluster")
    rfdetr_botsort = create_vision_system("rfdetr", "botsort", "bev_cluster")

    assert type(rfdetr_deepsort.detector).__name__ == "RFDETRDetector"
    assert type(rfdetr_deepsort.tracker).__name__ == "DeepSORTSingleCameraTracker"
    assert type(rfdetr_bytetrack.detector).__name__ == "RFDETRDetector"
    assert type(rfdetr_bytetrack.tracker).__name__ == "ByteTrackSingleCameraTracker"
    assert type(rfdetr_botsort.detector).__name__ == "RFDETRDetector"
    assert type(rfdetr_botsort.tracker).__name__ == "BoTSORTSingleCameraTracker"


def test_vision_api_orchestrates_split_components():
    calls: list[str] = []

    class FakeDetector(VisionDetector):
        def __init__(self):
            super().__init__(BaseDetectorConfig())

        def detect(self, frame, camera_id):
            calls.append(f"detect:{camera_id}")
            return [Detection(bbox=(1, 2, 3, 4), confidence=0.9, class_name="person", class_id=1)]

        def update_config(self, config_update):
            pass

        def get_config_schema(self):
            return {}

        def get_statistics(self):
            return {"detector_type": "FakeDetector"}

    class FakeTracker(SingleCameraTracker):
        def __init__(self):
            super().__init__(BaseTrackerConfig())

        def track(self, detections, camera_id, timestamp, frame=None):
            calls.append(f"track:{camera_id}")
            return [Track(track_id=f"cam{camera_id}_track_1", bbox=(1, 2, 3, 4), confidence=0.9, age=1, camera_id=camera_id)]

        def get_config_schema(self):
            return {}

        def update_config(self, config_update):
            pass

        def get_statistics(self):
            return {"tracker_type": "FakeTracker"}

    class FakeBEVTransformer:
        def transform_to_bev(self, tracks):
            calls.append("bev")
            return [
                BEVTrack(
                    track_id=track.track_id,
                    bev_x=10.0,
                    bev_y=20.0,
                    confidence=track.confidence,
                    camera_id=track.camera_id,
                )
                for track in tracks
            ]

    class FakeMerger:
        def merge(self, bev_tracks, timestamp, stream_frames=None, reid_features=None):
            calls.append("merge")
            return bev_tracks

        def get_statistics(self):
            return {}

    api = VisionAPI(
        config=VisionSystemConfig(detector_type="dummy", tracker_type="dummy", merger_type="bev_cluster"),
        detector=FakeDetector(),
        tracker=FakeTracker(),
        bev_transformer=FakeBEVTransformer(),
        merger=FakeMerger(),
    )
    api.enable_tracking()

    result = api.process_combined_frame(np.zeros((480, 720, 3), dtype=np.uint8), timestamp=1.0, num_streams=1)

    assert result is not None
    assert len(result.all_stream_detections[0]) == 1
    assert len(result.all_stream_tracks[0]) == 1
    assert len(result.bev_tracks) == 1
    assert calls == ["detect:0", "track:0", "bev", "merge"]
