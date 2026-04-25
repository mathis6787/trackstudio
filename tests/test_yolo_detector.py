from __future__ import annotations

import builtins
import importlib
import sys
import types

import numpy as np

from trackstudio.core.vision_api import VisionAPI
from trackstudio.detector_factory import create_detector
from trackstudio.detectors.base import VisionDetector
from trackstudio.mergers.base import VisionMerger
from trackstudio.trackers.base import BaseTrackerConfig, SingleCameraTracker
from trackstudio.vision_config import VisionSystemConfig, YOLODetectorConfig


class FakeTensor:
    def __init__(self, data):
        self.data = np.array(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10, 20, 40, 80], [1, 2, 5, 10]])
        self.conf = FakeTensor([0.91, 0.95])
        self.cls = FakeTensor([0, 2])

    def __len__(self):
        return 2


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person", 2: "car"}


def _load_yolo_module(monkeypatch, fake_yolo_class):
    sys.modules.pop("trackstudio.detectors.yolo", None)
    ultralytics_module = types.ModuleType("ultralytics")
    ultralytics_module.YOLO = fake_yolo_class
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)
    return importlib.import_module("trackstudio.detectors.yolo")


def test_yolo_config_exposes_all_yolo26_detection_weights():
    config = VisionSystemConfig(detector_type="yolo", tracker_type="dummy", merger_type="bev_cluster")
    yolo_config = config.get_detector_config()
    schema = yolo_config.model_json_schema()
    options = schema["$defs"]["YOLOModelConfig"]["properties"]["weights"]["options"]

    assert type(yolo_config).__name__ == "YOLODetectorConfig"
    assert config.get_available_detectors() == ["rfdetr", "yolo", "dummy"]
    assert options == ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]


def test_yolo_detector_converts_ultralytics_person_boxes(monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(("init", weights))

        def predict(self, frame, **kwargs):
            calls.append(("predict", kwargs))
            return [FakeResult()]

    yolo_module = _load_yolo_module(monkeypatch, FakeYOLO)
    config = YOLODetectorConfig()
    config.model.device = "cpu"
    detector = yolo_module.YOLODetector(config)

    detections = detector.detect(np.zeros((100, 120, 3), dtype=np.uint8), camera_id=0)

    assert calls[0] == ("init", "yolo26n.pt")
    assert calls[1][1]["imgsz"] == 640
    assert calls[1][1]["classes"] == [0]
    assert calls[1][1]["device"] == "cpu"
    assert len(detections) == 1
    assert detections[0].bbox == (10, 20, 30, 60)
    assert detections[0].confidence == 0.91
    assert detections[0].class_name == "person"
    assert detections[0].class_id == 0


def test_yolo_detector_can_emit_all_classes(monkeypatch):
    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, frame, **kwargs):
            assert kwargs["classes"] is None
            return [FakeResult()]

    yolo_module = _load_yolo_module(monkeypatch, FakeYOLO)
    config = YOLODetectorConfig()
    config.model.device = "cpu"
    config.model.person_only = False
    config.detection.min_box_width = 1
    config.detection.min_box_height = 1
    detector = yolo_module.YOLODetector(config)

    detections = detector.detect(np.zeros((100, 120, 3), dtype=np.uint8), camera_id=0)

    assert [d.class_name for d in detections] == ["person", "car"]


def test_yolo_factory_creates_detector_with_stubbed_dependency(monkeypatch):
    class FakeYOLO:
        def __init__(self, weights):
            pass

    _load_yolo_module(monkeypatch, FakeYOLO)
    config = VisionSystemConfig(detector_type="yolo", tracker_type="dummy", merger_type="bev_cluster")

    detector = create_detector(config)

    assert type(detector).__name__ == "YOLODetector"


def test_yolo_missing_dependency_error_is_actionable(monkeypatch):
    sys.modules.pop("trackstudio.detectors.yolo", None)
    sys.modules.pop("ultralytics", None)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ultralytics"):
            raise ImportError("No module named ultralytics")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    config = VisionSystemConfig(detector_type="yolo", tracker_type="dummy", merger_type="bev_cluster")

    try:
        create_detector(config)
    except ImportError as exc:
        assert "uv sync" in str(exc)
    else:
        raise AssertionError("Expected missing Ultralytics dependency to fail clearly")


def test_nested_yolo_config_updates_preserve_sibling_model_fields():
    class FakeDetector(VisionDetector):
        def __init__(self):
            super().__init__(YOLODetectorConfig())
            self.last_update = None

        def detect(self, frame, camera_id):
            return []

        def update_config(self, config_update):
            self.last_update = config_update

        def get_config_schema(self):
            return {}

        def get_statistics(self):
            return {}

    class FakeTracker(SingleCameraTracker):
        def __init__(self):
            super().__init__(BaseTrackerConfig())

        def track(self, detections, camera_id, timestamp, frame=None):
            return []

        def get_config_schema(self):
            return {}

        def update_config(self, config_update):
            pass

        def get_statistics(self):
            return {}

    class FakeBEVTransformer:
        def transform_to_bev(self, tracks):
            return []

    class FakeMerger(VisionMerger):
        def merge(self, bev_tracks, timestamp, stream_frames=None, reid_features=None):
            return []

        def get_statistics(self):
            return {}

    detector = FakeDetector()
    config = VisionSystemConfig(detector_type="yolo", tracker_type="dummy", merger_type="bev_cluster")
    api = VisionAPI(
        config=config,
        detector=detector,
        tracker=FakeTracker(),
        bev_transformer=FakeBEVTransformer(),
        merger=FakeMerger(),
    )

    api.update_config({"yolo_detector": {"model": {"weights": "yolo26s.pt"}}})
    api.update_config({"yolo_detector": {"model": {"image_size": 768}}})

    assert api.config.yolo_detector.model.weights == "yolo26s.pt"
    assert api.config.yolo_detector.model.image_size == 768
    assert detector.last_update["model"]["weights"] == "yolo26s.pt"
