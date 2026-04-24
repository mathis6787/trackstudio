# Tracking Behavior & Configuration

## How Tracking Works

TrackStudio now separates detection from tracking:

```json
{
  "detector_type": "rfdetr",
  "tracker_type": "deepsort",
  "merger_type": "bev_cluster"
}
```

`detector_type` chooses the object detector. Today the real detector is `rfdetr`.

`tracker_type` chooses the single-camera tracking algorithm. Today the real trackers are:

- `deepsort`
- `bytetrack`

The vision pipeline runs independently of the source stream FPS:

```text
Source video (any fps) -> decoder -> detector -> tracker -> BEV transformer -> merger
```

The source video FPS only affects the smoothness of the video in the browser. Tracking timing is based on `vision_fps`, which defaults to `10`.

Changing the tracker in the UI updates the config immediately, but the active tracker object is only replaced after **Restart System**. Restarting resets track state and track IDs.

---

## DeepSORT vs ByteTrack

DeepSORT and ByteTrack do not expose the same settings because they associate detections differently.

DeepSORT uses:

- motion prediction
- bounding box overlap
- appearance/ReID features from a person crop

ByteTrack uses:

- motion prediction
- bounding box overlap
- detection confidence, including lower-confidence detections for re-association

ByteTrack in this project does **not** use ReID appearance features. That means DeepSORT settings such as `Max Cosine Distance` and `ReID Matching Threshold` are not relevant for ByteTrack.

---

## ByteTrack Settings

These are the ByteTrack settings shown in the UI when `tracker_type` is `bytetrack`.

| UI label | Config field | Default | Meaning |
|---|---:|---:|---|
| Activation Threshold | `track_activation_threshold` | `0.25` | Minimum detection confidence required to start a new track. Existing tracks can still be matched with lower-confidence detections. |
| Lost Track Buffer (frames) | `lost_track_buffer` | `50` | Number of vision frames to keep a lost track alive without a matching detection. |
| IoU Matching Threshold | `minimum_matching_threshold` | `0.8` | Box-overlap threshold used for matching detections to existing tracks. Lower is more lenient; higher is stricter. |
| Frame Rate | `frame_rate` | `10` | FPS used by ByteTrack's Kalman predictor. This should usually match `vision_fps`. |
| Min Consecutive Frames | `minimum_consecutive_frames` | `1` | Number of consecutive frames a track must appear before it is reported. |

### Occlusion Tolerance

For ByteTrack, the closest equivalent to DeepSORT's `tracker_max_age` is:

```text
lost_track_buffer
```

The real-world tolerance is:

```text
tolerance_seconds = lost_track_buffer / vision_fps
```

Examples at `vision_fps = 10`:

| lost_track_buffer | Occlusion tolerance |
|---:|---:|
| 30 | 3 seconds |
| 50 (default) | 5 seconds |
| 100 | 10 seconds |
| 150 | 15 seconds |

For retail scenes with shelves and displays, `80-150` is a reasonable starting range if people frequently disappear behind fixtures. Higher values preserve IDs longer, but can also keep stale tracks alive longer after someone leaves.

### Activation Threshold

`track_activation_threshold` controls how confident a detection must be before ByteTrack starts a new ID.

Lower values:

- start tracks more easily
- can recover harder detections
- may create more false tracks

Higher values:

- reduce false tracks
- can miss people with low detector confidence

A practical range is `0.20-0.40`.

### IoU Matching Threshold

`minimum_matching_threshold` controls how much a detection must overlap with the predicted track box.

Lower values are more forgiving when motion is fast or boxes jitter. Higher values are stricter and can reduce incorrect matches when people are close together.

A practical range is `0.6-0.9`.

### Frame Rate

`frame_rate` should usually match `vision_fps`.

If `vision_fps` is `10`, keep ByteTrack `frame_rate` at `10`. If you increase `vision_fps` to `15`, also change ByteTrack `frame_rate` to `15`.

---

## DeepSORT Settings

These settings are only relevant when `tracker_type` is `deepsort`.

| UI label | Config field | Default | Meaning |
|---|---:|---:|---|
| Max Track Age | `tracker_max_age` | `30` | Number of vision frames to keep a track alive without detection. |
| Min Hits to Start | `tracker_min_hits` | `1` | Consecutive detections required before a track is trusted. |
| Max IoU Distance | `tracker_max_iou_distance` | `0.7` | Box-overlap matching threshold. |
| Max Cosine Distance | `tracker_max_cosine_distance` | `0.3` | Appearance/ReID distance threshold. |
| ReID Matching Threshold | `tracker_matching_threshold` | `0.3` | ReID feature matching threshold. |

For DeepSORT, occlusion tolerance is:

```text
tolerance_seconds = tracker_max_age / vision_fps
```

Examples at `vision_fps = 10`:

| tracker_max_age | Occlusion tolerance |
|---:|---:|
| 30 (default) | 3 seconds |
| 60 | 6 seconds |
| 100 | 10 seconds |
| 150 | 15 seconds |

---

## Settings That Do Not Map 1:1

Some settings have rough equivalents:

| DeepSORT | ByteTrack |
|---|---|
| `tracker_max_age` | `lost_track_buffer` |
| `tracker_min_hits` | `minimum_consecutive_frames` |
| `tracker_max_iou_distance` | `minimum_matching_threshold` |

Some DeepSORT settings have no ByteTrack equivalent:

- `tracker_max_cosine_distance`
- `tracker_matching_threshold`

Those are ReID/appearance settings, and ByteTrack does not use ReID in this implementation.

---

## How to Change Tracker Settings

### UI

Open the **Vision Processor Controls** panel.

1. Select the tracker: `DeepSORT` or `ByteTrack`.
2. Adjust the settings shown for that tracker.
3. Click **Restart System** if you changed the tracker algorithm.

Changing numeric settings updates the runtime config. Changing the tracker algorithm itself requires restart because it creates a different tracker class with fresh internal state.

### Source Defaults

Defaults live in `trackstudio/vision_config.py`:

- `DeepSORTConfig`
- `ByteTrackConfig`

Example ByteTrack default:

```python
lost_track_buffer: int = int_slider_field(
    50,
    5,
    300,
    5,
    "Lost Track Buffer (frames)",
    "Frames to keep a track alive without a matching detection.",
)
```

---

## ReID and Re-identification

DeepSORT uses an OSNet appearance model to help match a person based on visual appearance.

ByteTrack does not use ReID in this project. It can maintain an ID through short occlusions using motion and low-confidence detections, but if a person disappears for longer than `lost_track_buffer`, they will normally receive a new ID when they reappear.

ReID reliability depends on:

- clothing similarity
- camera angle and resolution
- lighting conditions
- how much of the person is visible
