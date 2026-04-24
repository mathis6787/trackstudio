from __future__ import annotations

import builtins
import importlib
import sys
import types

import numpy as np

from trackstudio.tracker_factory import create_tracker
from trackstudio.vision_config import BoTSORTConfig, BoTSORTTrackerConfig, VisionSystemConfig
from trackstudio.vision_types import Detection, Track


def _load_botsort_module(monkeypatch, fake_bot_sort_class):
    sys.modules.pop("trackstudio.trackers.botsort", None)
    sys.modules.pop("boxmot.trackers.botsort.botsort", None)
    boxmot_module = types.ModuleType("boxmot")
    boxmot_module.__path__ = []
    trackers_module = types.ModuleType("boxmot.trackers")
    trackers_module.__path__ = []
    botsort_package = types.ModuleType("boxmot.trackers.botsort")
    botsort_package.__path__ = []
    botsort_module = types.ModuleType("boxmot.trackers.botsort.botsort")
    botsort_module.BotSort = fake_bot_sort_class

    monkeypatch.setitem(sys.modules, "boxmot", boxmot_module)
    monkeypatch.setitem(sys.modules, "boxmot.trackers", trackers_module)
    monkeypatch.setitem(sys.modules, "boxmot.trackers.botsort", botsort_package)
    monkeypatch.setitem(sys.modules, "boxmot.trackers.botsort.botsort", botsort_module)
    return importlib.import_module("trackstudio.trackers.botsort")


def test_botsort_missing_dependency_error_is_actionable(monkeypatch):
    sys.modules.pop("trackstudio.trackers.botsort", None)
    sys.modules.pop("boxmot", None)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("boxmot"):
            raise ImportError("No module named boxmot")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    config = VisionSystemConfig(detector_type="dummy", tracker_type="botsort", merger_type="bev_cluster")
    try:
        create_tracker(config)
    except ImportError as exc:
        message = str(exc)
        assert "uv sync" in message
    else:
        raise AssertionError("Expected missing BoxMOT dependency to fail clearly")


def test_botsort_config_rejects_legacy_with_reid_name():
    try:
        BoTSORTConfig(with_reid=False)
    except ValueError as exc:
        assert "with_reid" in str(exc)
    else:
        raise AssertionError("Expected old with_reid config key to be rejected")


def test_botsort_rejects_unknown_reid_backend(monkeypatch):
    class FakeBotSort:
        pass

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)
    config = BoTSORTTrackerConfig(
        tracking=BoTSORTConfig(use_reid_matching=False, reid_backend="boxmot")
    )

    try:
        botsort_module.BoTSORTSingleCameraTracker(config)
    except ValueError as exc:
        assert "Unsupported BoT-SORT ReID backend" in str(exc)
    else:
        raise AssertionError("Expected unsupported ReID backend to fail clearly")


def test_botsort_tracker_converts_detections_and_tracks(monkeypatch):
    calls = []

    class FakeBotSort:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def update(self, dets, img, embs=None):
            calls.append(("update", dets.copy(), img.shape, embs))
            if len(dets) == 0:
                return np.empty((0, 8), dtype=np.float32)
            return np.array([[10, 20, 40, 80, 7, 0.91, 1, 0]], dtype=np.float32)

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)
    config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=False))
    tracker = botsort_module.BoTSORTSingleCameraTracker(config)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    no_tracks = tracker.track([], camera_id=2, timestamp=1.0, frame=frame)
    tracks = tracker.track(
        [Detection(bbox=(10, 20, 30, 60), confidence=0.91, class_name="person", class_id=1)],
        camera_id=2,
        timestamp=2.0,
        frame=frame,
    )

    assert no_tracks == []
    np.testing.assert_array_equal(calls[1][1], np.empty((0, 6), dtype=np.float32))
    assert calls[1][3] is None
    np.testing.assert_array_equal(calls[2][1], np.array([[10, 20, 40, 80, 0.91, 1]], dtype=np.float32))
    assert calls[2][3] is None
    assert tracks[0].track_id == "cam2_track_7"
    assert tracks[0].bbox == (10, 20, 30, 60)
    assert abs(tracks[0].confidence - 0.91) < 0.001
    assert tracks[0].camera_id == 2


def test_botsort_update_config_clears_existing_trackers(monkeypatch):
    class FakeBotSort:
        def __init__(self, **kwargs):
            pass

        def update(self, dets, img, embs=None):
            return np.empty((0, 8), dtype=np.float32)

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)
    config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=False))
    tracker = botsort_module.BoTSORTSingleCameraTracker(config)
    tracker.track([], camera_id=0, timestamp=1.0, frame=np.zeros((20, 20, 3), dtype=np.uint8))

    assert 0 in tracker.trackers

    tracker.update_config({"tracking": {"use_reid_matching": False, "track_buffer": 100}})

    assert tracker.trackers == {}
    assert tracker.config.tracking.track_buffer == 100


def test_botsort_passes_reid_embeddings_to_boxmot(monkeypatch):
    calls = []

    class FakeBotSort:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def update(self, dets, img, embs=None):
            calls.append(("update", dets.copy(), embs.copy()))
            return np.empty((0, 8), dtype=np.float32)

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)
    config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=False))
    tracker = botsort_module.BoTSORTSingleCameraTracker(config)
    tracker.config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=True))

    class FakeAdapter:
        def get_features(self, xyxy_boxes, frame):
            np.testing.assert_array_equal(xyxy_boxes, np.array([[10, 20, 40, 80]], dtype=np.float32))
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    tracker.reid_adapter = FakeAdapter()
    tracker.track(
        [Detection(bbox=(10, 20, 30, 60), confidence=0.91, class_name="person", class_id=1)],
        camera_id=2,
        timestamp=2.0,
        frame=np.zeros((100, 120, 3), dtype=np.uint8),
    )

    assert calls[0][1]["with_reid"] is False
    np.testing.assert_array_equal(calls[1][2], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))


def test_botsort_passes_empty_embeddings_when_reid_has_no_detections(monkeypatch):
    calls = []

    class FakeBotSort:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def update(self, dets, img, embs=None):
            calls.append(("update", dets.copy(), embs.copy() if embs is not None else None))
            return np.empty((0, 8), dtype=np.float32)

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)
    config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=False))
    tracker = botsort_module.BoTSORTSingleCameraTracker(config)
    tracker.config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=True))
    tracker.reid_adapter = object()

    tracks = tracker.track([], camera_id=2, timestamp=2.0, frame=np.zeros((100, 120, 3), dtype=np.uint8))

    assert tracks == []
    np.testing.assert_array_equal(calls[1][2], np.empty((0, 0), dtype=np.float32))


def test_botsort_reid_adapter_returns_embeddings(monkeypatch):
    class FakeBotSort:
        pass

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)

    class FakeExtractor:
        def extract_features(self, frame, detections):
            assert frame.shape == (20, 20, 3)
            np.testing.assert_array_equal(detections, np.array([[1, 2, 3, 4]], dtype=np.float32))
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    adapter = botsort_module.BoxMOTTorchReIDAdapter(FakeExtractor())
    features = adapter.get_features(
        np.array([[1, 2, 3, 4]], dtype=np.float32),
        np.zeros((20, 20, 3), dtype=np.uint8),
    )

    np.testing.assert_array_equal(features, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))


def test_botsort_get_reid_features_uses_adapter(monkeypatch):
    class FakeBotSort:
        pass

    botsort_module = _load_botsort_module(monkeypatch, FakeBotSort)

    class FakeAdapter:
        def get_features(self, xyxy_boxes, frame):
            assert frame.shape == (20, 20, 3)
            np.testing.assert_array_equal(xyxy_boxes, np.array([[2, 3, 12, 23]], dtype=np.float32))
            return np.array([[0.4, 0.5]], dtype=np.float32)

    config = BoTSORTTrackerConfig(tracking=BoTSORTConfig(use_reid_matching=False))
    tracker = botsort_module.BoTSORTSingleCameraTracker(config)
    tracker.reid_adapter = FakeAdapter()

    features = tracker.get_reid_features(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [Track(track_id="cam0_track_1", bbox=(2, 3, 10, 20), confidence=0.9, age=1, camera_id=0)],
    )

    np.testing.assert_array_equal(features, np.array([[0.4, 0.5]], dtype=np.float32))
