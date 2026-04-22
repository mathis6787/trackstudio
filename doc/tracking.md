# Tracking Behavior & Configuration

## How Tracking Works

TrackStudio uses **RF-DETR** for detection and **DeepSORT** for tracking.

The vision pipeline runs independently of the source stream FPS:

```
Source video (any fps) → decoder → vision pipeline (vision_fps) → DeepSORT
```

DeepSORT only receives frames at the `vision_fps` rate. The source video FPS only affects the smoothness of the video in the browser — it has **no effect on tracking quality or ID stability**.

---

## Occlusion & ID Changes

When someone is hidden behind a shelf or display, the detector can no longer see them. DeepSORT keeps the track alive for `tracker_max_age` frames. If they remain hidden longer than that, the track is deleted and a **new ID is assigned** when they reappear.

This is expected behavior — not a bug.

---

## tracker_max_age

`tracker_max_age` is the number of vision pipeline frames DeepSORT waits before deleting a track with no detections.

### Formula

$$\text{tolerance (seconds)} = \frac{\text{tracker\_max\_age}}{\text{vision\_fps}}$$

### Examples at vision_fps = 10

| tracker_max_age | Occlusion tolerance |
|---|---|
| 30 (default) | 3 seconds |
| 60 | 6 seconds |
| 100 | 10 seconds |
| 150 | 15 seconds |

### Retail Store Recommendation

For retail environments with shelving and displays, **80–150** is a good range. This handles typical browse durations behind a display without creating too many "ghost tracks" (IDs that linger after someone has left).

---

## Source Video FPS

The source stream FPS does **not** affect `max_age` calculations. The formula always uses `vision_fps`.

| Source FPS | vision_fps | max_age | Tolerance |
|---|---|---|---|
| 15 fps | 10 | 30 | 3 seconds |
| 30 fps | 10 | 30 | 3 seconds |

Only changing `vision_fps` would require adjusting `max_age` to maintain the same real-world tolerance.

---

## How to Change tracker_max_age

### Option 1 — UI (does not persist across restarts)

In the web UI at http://127.0.0.1:8000, open the **Vision Processor Controls** panel and drag the **Max Track Age** slider.

### Option 2 — Source code default (persists)

Edit `trackstudio/vision_config.py`, line ~97:

```python
# Change the first argument (30) to your desired default
tracker_max_age: int = int_slider_field(30, 5, 200, 1, "Max Track Age", "Frames to keep a track without detection.")
```

> Note: `tracker_max_age` cannot currently be set via the JSON config file — it is not forwarded by the CLI.

---

## Memory Usage

A higher `tracker_max_age` means more lost tracks can be simultaneously alive in memory. Each lost track holds:
- Kalman filter state (a few floats)
- ReID appearance feature vector (~512 float32 values ≈ 2 KB)
- Bounding box history

In a typical retail store with ~10 people, even if all are simultaneously occluded, that is ~20 KB of extra memory — completely irrelevant.

Memory only becomes a concern with hundreds of simultaneously lost tracks (e.g. extremely crowded scenes), and even then the impact is MB-level, not GB-level.

**Do not factor RAM into your `tracker_max_age` decision.** Set it based purely on how long people typically disappear behind shelves.

---

## ReID and Re-identification

DeepSORT uses an **OSNet** appearance model (ReID) to try to re-match a person even after their track has been deleted. If someone reappears and looks visually similar, the same ID may be reassigned.

In practice, from ceiling-mounted retail cameras, ReID reliability varies based on:
- Clothing similarity
- Camera angle and resolution
- Lighting conditions
