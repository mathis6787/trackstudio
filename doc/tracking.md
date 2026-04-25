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

`detector_type` chooses the object detector. The real detectors are:

- `rfdetr`
- `yolo`

The `yolo` detector currently supports YOLO26 detection checkpoints:

```text
yolo26n.pt
yolo26s.pt
yolo26m.pt
yolo26l.pt
yolo26x.pt
```

Use the `yolo_detector.model.weights` setting to switch between them. `n` is the fastest/lightest model and `x` is the slowest/heaviest model.

`tracker_type` chooses the single-camera tracking algorithm. Today the real trackers are:

- `deepsort`
- `bytetrack`
- `botsort`

BoT-SORT is implemented as a separate tracker component so it can consume detections from any detector, for example:

```text
rfdetr -> botsort
yolo   -> botsort
```

If YOLO later becomes both the primary detector and the production tracking path, it may be worth adding a separate YOLO-native fast path that uses Ultralytics' built-in BoT-SORT integration. That path should be treated as an optimized YOLO-specific mode, not as the default TrackStudio tracker abstraction.

TrackStudio only supports YOLO26 detection models in the detector interface right now. Segmentation, pose, OBB, and classification checkpoints are intentionally not wired into tracking yet because the current trackers consume axis-aligned detection boxes.

The vision pipeline runs independently of the source stream FPS:

```text
Source video (any fps) -> decoder -> detector -> tracker -> BEV transformer -> merger
```

The source video FPS only affects the smoothness of the video in the browser. Tracking timing is based on `vision_fps`, which defaults to `10`.

Changing the tracker in the UI updates the config immediately, but the active tracker object is only replaced after **Restart System**. Restarting resets track state and track IDs.

---

## DeepSORT vs ByteTrack vs BoT-SORT

DeepSORT, ByteTrack, and BoT-SORT do not expose the same settings because they associate detections differently.

DeepSORT uses:

- motion prediction
- bounding box overlap
- appearance/ReID features from a person crop

ByteTrack uses:

- motion prediction
- bounding box overlap
- detection confidence, including lower-confidence detections for re-association

ByteTrack in this project does **not** use ReID appearance features. That means DeepSORT settings such as `Max Cosine Distance` and `ReID Matching Threshold` are not relevant for ByteTrack.

BoT-SORT is similar to ByteTrack but can add stronger association logic, optional appearance/ReID matching, and optional camera motion compensation. In TrackStudio, the clean implementation should be detector-agnostic: it should accept normalized `Detection` objects from RF-DETR, YOLO, or any future detector.

If the detector is specifically YOLO and the goal is a YOLO-only optimized production path, Ultralytics' native `model.track(..., tracker="botsort.yaml")` flow may be better integrated. That should be documented and implemented as a YOLO-specific shortcut if needed, because it couples detection and tracking again.

---

## Detector Model vs Tracker Parameters

Tracker parameters are not fully independent from the detector model. Trackers do not start from raw video pixels. They mostly consume detector output:

```text
bbox + confidence + class_id
```

Changing from `yolo26n.pt` to `yolo26l.pt`, or from YOLO to RF-DETR, can change:

- how many people are detected
- how often detections are missed
- how stable the boxes are from frame to frame
- how confidence scores are distributed
- how many false positives appear

Because of that, tracker settings that work well with one detector are a good starting point for another detector, but they are not guaranteed to stay optimal.

### YOLO Model Size

YOLO26 model size affects tracker behavior:

| Model | Expected detector behavior | Tracker impact |
|---|---|---|
| `yolo26n.pt` | Fastest, lightest, more likely to miss hard people or produce lower confidence | May need lower thresholds, longer lost-track buffers, and more tolerant matching. |
| `yolo26s.pt` / `yolo26m.pt` | Middle ground | Usually the best place to start for tuning. |
| `yolo26l.pt` / `yolo26x.pt` | Slower, stronger detections, often higher confidence and better boxes | Can often use stricter thresholds, but processing latency may increase. |

If the video already runs slowly, do not assume a larger model is better. Better detection accuracy only helps if the system can still process frames fast enough for the scene.

### RF-DETR vs YOLO

RF-DETR and YOLO can both detect people, but they may produce different box shapes, confidence scores, and miss/false-positive patterns. A ByteTrack or BoT-SORT config tuned on YOLO should be retested before using it with RF-DETR.

Treat tuned parameters as tied to this combination:

```text
detector_type + detector weights + detector thresholds + tracker_type + tracker thresholds + vision_fps
```

Example:

```text
yolo + yolo26s.pt + confidence_threshold=0.25 + bytetrack + vision_fps=10
```

That is a different tuning profile from:

```text
yolo + yolo26l.pt + confidence_threshold=0.35 + bytetrack + vision_fps=10
```

and also different from:

```text
rfdetr + confidence_threshold=0.25 + bytetrack + vision_fps=10
```

### Most Detector-Sensitive Settings

Start by tuning detector settings first:

| Config field | Why it matters |
|---|---|
| `confidence_threshold` | Controls which detections reach the tracker. Too high causes missed people; too low can create false tracks. |
| `nms_iou_threshold` | Controls duplicate box suppression. Bad NMS can create duplicate people or remove valid close people. |
| `min_box_width` / `min_box_height` | Filters tiny detections. Useful for noise, but can remove distant people. |
| `max_aspect_ratio` | Filters unusual boxes. Useful for bad detections, but can remove bent/partial people. |
| `yolo_detector.model.image_size` | Larger image size can improve small/distant people, but increases latency. |

Then tune tracker settings:

| Tracker | Sensitive fields |
|---|---|
| ByteTrack | `track_activation_threshold`, `lost_track_buffer`, `minimum_matching_threshold`, `frame_rate` |
| BoT-SORT | `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `proximity_thresh`, `appearance_thresh`, `frame_rate` |
| DeepSORT | `tracker_max_age`, `tracker_min_hits`, `tracker_max_iou_distance`, `tracker_max_cosine_distance`, `tracker_matching_threshold` |

### Recommended Test Order

1. Choose one detector and one model weight.
   Start with `yolo26s.pt` or `yolo26m.pt` if you are testing YOLO. Use `yolo26n.pt` if speed is the main constraint.

2. Tune detector quality before tracker quality.
   Watch raw detections. Fix missed people, duplicate boxes, and obvious false positives before changing tracker settings.

3. Tune one tracker at a time.
   For ByteTrack, start with `track_activation_threshold`, `minimum_matching_threshold`, and `lost_track_buffer`.

4. Keep `vision_fps` and tracker `frame_rate` aligned.
   If `vision_fps = 10`, keep ByteTrack/BoT-SORT `frame_rate = 10`.

5. Save tuning profiles by detector/model.
   Do not assume one profile is universal. Name your test notes like `yolo26s_bytetrack_store_daylight` or `rfdetr_botsort_aisle_occlusion`.

6. Retest when changing detector family, YOLO weight, image size, or confidence threshold.
   A full retune may not be needed, but validation is required.

### Practical Rule

If you found good ByteTrack settings for your environment with `yolo26n.pt`, use them as the first baseline for `yolo26s.pt` or `yolo26l.pt`, but verify the results again. If you switch to RF-DETR, use the old settings only as a rough starting point.

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

## BoT-SORT Settings

BoT-SORT uses BoxMOT, which is installed by the normal project sync:

```bash
uv sync
```

TrackStudio uses RF-DETR `1.6+`, which resolves with the newer Hugging Face stack required by BoxMOT 18. Avoid downgrading RF-DETR or pinning Transformers back to 4.x if you want BoT-SORT to keep working through `uv sync`.

BoxMOT is AGPL-3.0 licensed. Revisit that license before selling or redistributing a product that includes BoT-SORT/BoxMOT.

These are the BoT-SORT settings shown in the UI when `tracker_type` is `botsort`.

| UI label | Config field | Default | Meaning |
|---|---:|---:|---|
| High Confidence Threshold | `track_high_thresh` | `0.5` | Detection confidence threshold for first association. |
| Low Confidence Threshold | `track_low_thresh` | `0.1` | Lower confidence bound for second-stage candidate detections. |
| New Track Threshold | `new_track_thresh` | `0.6` | Confidence required to initialize a new track. |
| Lost Track Buffer (frames) | `track_buffer` | `50` | Frames to keep an unmatched track alive. |
| Matching Threshold | `match_thresh` | `0.8` | Association threshold for matching tracks to detections. |
| Proximity Threshold | `proximity_thresh` | `0.5` | IoU gate used before appearance matching. |
| Appearance Threshold | `appearance_thresh` | `0.25` | Maximum embedding distance accepted for ReID matching. |
| Use ReID Matching | `use_reid_matching` | `true` | Use TrackStudio ReID embeddings for BoT-SORT appearance association. |
| ReID Backend | `reid_backend` | `trackstudio` | Source of ReID embeddings. Currently only TrackStudio's shared TorchReID/OSNet backend is implemented. |
| Camera Motion Compensation | `cmc_method` | `none` | Use `none` for fixed cameras. Moving cameras can try methods supported by BoxMOT. |
| Frame Rate | `frame_rate` | `10` | FPS used by BoT-SORT track buffer scaling. Match `vision_fps`. |
| Fuse First Association | `fuse_first_associate` | `false` | Fuse motion and appearance in the first association step. |

For fixed retail/security cameras, keep `cmc_method` set to `none`. Camera motion compensation is useful for moving cameras, but wastes compute and can add failure modes for static cameras.

TrackStudio currently feeds BoT-SORT with the existing TorchReID/OSNet extractor so DeepSORT, BoT-SORT, and cross-camera merging share one appearance pipeline. In production, if BoT-SORT becomes the primary tracker, it may be better to use BoxMOT's native ReID backend/model loading instead. That choice should be benchmarked against the shared TrackStudio ReID path for accuracy, memory use, startup time, and deployment packaging.

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

1. Select the tracker: `DeepSORT`, `ByteTrack`, or `BoT-SORT`.
2. Adjust the settings shown for that tracker.
3. Click **Restart System** if you changed the tracker algorithm.

Changing numeric settings updates the runtime config. Changing the tracker algorithm itself requires restart because it creates a different tracker class with fresh internal state.

### Source Defaults

Defaults live in `trackstudio/vision_config.py`:

- `DeepSORTConfig`
- `ByteTrackConfig`
- `BoTSORTConfig`

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

BoT-SORT also supports ReID. In this project, BoT-SORT currently reuses the same TorchReID/OSNet extractor as DeepSORT instead of loading a second ReID model through BoxMOT. For a production BoT-SORT-first deployment, using BoxMOT's own ReID backend may be a reasonable alternative if it is faster or easier to package.

ByteTrack does not use ReID in this project. It can maintain an ID through short occlusions using motion and low-confidence detections, but if a person disappears for longer than `lost_track_buffer`, they will normally receive a new ID when they reappear.

ReID reliability depends on:

- clothing similarity
- camera angle and resolution
- lighting conditions
- how much of the person is visible

### Anti-Shoplifting Behavior Systems

For anti-shoplifting, a single-camera ID reset is not automatically catastrophic, but it depends where the reset happens in the pipeline.

If a person disappears behind a display and comes back with a new local track ID, the raw single-camera tracker lost continuity. That can hurt features that depend on a continuous per-person timeline:

- linger time
- path history
- hand-to-shelf movement sequence
- object interaction sequence
- pose sequence over time

This is less serious if a later stage can merge track fragments back together using signals such as:

- same camera
- close time gap
- close image or BEV position
- appearance/ReID similarity
- motion direction
- camera zone
- pose continuity
- object interaction continuity
- height or box-size consistency

Do not make an LSTM, XGBoost model, or other behavior classifier consume raw tracker IDs blindly. A better architecture is:

```text
detector / pose model
-> single-camera tracker
-> BEV transform
-> same-camera and cross-camera tracklet merging
-> customer session builder
-> behavior feature extraction
-> LSTM / XGBoost / rules / Bayesian layer
-> suspicious behavior score
```

The behavior model should usually run after tracklet merging or after a customer-session builder. Otherwise, every occlusion can split a customer's history into separate identities.

ReID failures are normal in retail scenes because shelves, displays, carts, similar clothing, camera angle changes, low-resolution crops, and lighting changes all reduce appearance reliability. Do not rely on ReID alone for customer identity. Combine several weak signals instead.

A Bayesian layer can help by keeping identity and behavior as probabilities instead of hard decisions. For example:

```text
P(same_customer | appearance, BEV distance, time gap, direction, zone)
```

Practical rule: occasional ID switches are acceptable if there is a session or tracklet merger after tracking. They are a big problem only if downstream behavior models treat every tracker ID as a final customer identity.

---

## Current Camera Merging

TrackStudio currently uses `bev_cluster` for multi-camera merging.

The current flow is:

```text
detector
-> single-camera tracker
-> BEV transform
-> BEV cluster merger
-> global_id assignment
```

Each camera first produces local tracks. For example:

```text
camera 0 local track 1
camera 1 local track 3
```

The BEV transformer converts each local track into bird's-eye-view coordinates using the bottom-center of the bounding box, which approximates the person's feet.

The merger then receives all BEV tracks from all cameras. It clusters tracks from different cameras only. It intentionally does not merge tracks from the same camera during this clustering step.

Two tracks from different cameras are considered the same person when:

- their BEV positions are close enough, based on `spatial_threshold`
- and, if ReID features are available, their appearance distance is below `appearance_threshold`

When tracks are clustered, the merger assigns a shared `global_id`. That means separate local camera tracks can become one global person identity.

### What Current Merging Is Good For

The current `bev_cluster` merger is useful for:

- approximate cross-camera identity
- merging people visible in overlapping camera views
- simple BEV-level global tracking
- debugging camera calibration and multi-camera alignment
- early testing of detector/tracker combinations

### Current Limitations

The current merger is not yet a full customer-session system.

It is weak for:

- reconnecting someone after a long occlusion
- reconnecting a same-camera ID switch
- building a durable customer session over minutes
- preserving behavior history through shelves, displays, carts, or crowds
- producing high-confidence identity continuity for suspicious-behavior analysis

If camera 0 loses a person behind a display and the tracker creates a new local ID, the current merger will usually create a new global ID too. It does not yet compare a new same-camera tracklet against old global tracks using time gap, BEV distance, direction, appearance, and zone context.

Also, the merger configuration currently exposes `appearance_weight`, `smoothing_alpha`, and `velocity_alpha`, but the current implementation is mostly hard spatial gating plus optional ReID gating. Those fields should be treated as future/improvement parameters until the merger uses them directly.

### Anti-Shoplifting Recommendation

For anti-shoplifting, the current merger is a good foundation, but it should be followed by a stronger customer-session builder:

```text
BEV cluster merger
-> same-camera tracklet reconnection
-> cross-camera session association
-> customer session timeline
-> behavior feature extraction
-> LSTM / XGBoost / rules / Bayesian layer
```

That session builder should combine several weak signals:

- BEV distance
- time gap
- motion direction
- camera zone
- appearance/ReID similarity
- pose continuity
- object-interaction continuity
- height or box-size consistency

For behavior detection, do not treat `global_id` as final truth. Treat it as the current best identity estimate. Downstream behavior models should be able to handle uncertain identity and fragmented tracklets.
