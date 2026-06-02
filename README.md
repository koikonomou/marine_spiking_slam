# Marine Spiking SLAM

A bio-inspired Simultaneous Localisation and Mapping (SLAM) system for unmanned surface vehicles (USVs), built on the [RatSLAM](https://github.com/davidmball/ratslam) algorithm and adapted for marine environments.

Obstacle detection and collision avoidance rely entirely on computer vision — no distance sensors (LiDAR, sonar, or ultrasonic) are required.

---

## Overview

The system runs a RatSLAM-style pipeline on a monocular camera stream:

```
Camera frame
    │
    ├─► Visual Odometry        — image profile matching → v_trans, v_rot
    ├─► Local View Matcher     — visual template matching → place ID (vt_id)
    ├─► Pose Cell Network      — 3-D attractor network (x, y, θ) → best pose
    ├─► Experience Map         — topological graph with spring relaxation → map
    │
    └─► Obstacle Detector      — Canny edges + blob filtering → obstacle mask + TTC
            │
            └─► Collision Avoider  — sector-based steering commands
```

### SLAM modules (inspired by RatSLAM)

| Module | File | Description |
|---|---|---|
| Visual Odometry | `src/visual_odometry.py` | Non-circular image profile shift matching to estimate translational and rotational velocity |
| Local View Matcher | `src/local_view.py` | Mean-normalised template database with fast brightness rejection and shift matching |
| Pose Cell Network | `src/pose_cells.py` | 3-D spiking attractor network with bilinear path integration and population-vector best-pose decoding |
| Experience Map | `src/experience_map.py` | Topological graph (nodes + links) with spring-based relaxation for loop-closure correction |

### Obstacle avoidance modules (vision-only)

| Module | File | Description |
|---|---|---|
| Obstacle Detector | `src/obstacle_detection.py` | Canny edge detection with proximity ROI and blob-size filtering to separate real obstacles from wave noise |
| Collision Avoider | `src/collision_avoidance.py` | Sector-based steering (left / centre / right) with emergency stop on low TTC |

---

## Installation

Requires [Poetry](https://python-poetry.org/).

```bash
git clone <repo-url>
cd marine_spiking_slam
poetry install
```

### Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing, Canny edges, connected components, display |
| `numpy` | Array operations throughout |
| `scipy` | Gaussian filter for pose cell attractor dynamics |
| `matplotlib` | Trajectory plotting in `eval.py` |

---

## Dataset

Tested on the [Maritime Visual Tracking Dataset (MVTD)](https://github.com/chenzx/MVTD).

Expected folder structure:

```
~/codes/datasets/Maritime_Visual_Tracking_Dataset_MVTD/
    train/
        119-USV/
            00000001.jpg
            00000002.jpg
            ...
            groundtruth.txt
```

To use a different sequence, update `dataset_path` at the top of `src/main.py` and `src/eval.py`.

---

## Usage

### Full pipeline (`main.py`)

```bash
poetry run marine-slam
```

Or directly:

```bash
poetry run python src/main.py
```

### Evaluation (`eval.py`)

```bash
poetry run python src/eval.py
```

Plots the estimated SLAM trajectory against ground-truth bounding-box centroids if a `groundtruth.txt` file is present alongside the frames.

---

## Visualisation

Three windows are shown during runtime:

| Window | Content |
|---|---|
| `Camera Feed + Avoidance HUD` | Live frame with obstacle pixels tinted red, sector pixel counts (L/C/R), time-to-collision, and current avoidance action |
| `Trajectory Map` | Top-down trajectory — cyan = safe nodes, red = danger nodes (obstacle observed) |
| `Obstacle Mask` | Binary mask of detected obstacle pixels |

Press **`q`** to quit.

---

## How obstacle detection works

Since no distance sensors are available, proximity and danger are estimated from computer vision alone.

### Detection pipeline

1. **Proximity ROI** — only the lower portion of the frame (from `PROXIMITY_FRAC` downward) is analysed. In a forward-looking marine camera, far objects sit near the horizon (top of the navigable zone) and move downward as the vessel approaches. Restricting the ROI avoids the sky and distant horizon clutter.

2. **Canny edge detection** — `Canny(85, 255)` extracts strong edges in the ROI. The high threshold (85) makes it selective: water surface texture and small waves produce gradients well below this level and are naturally suppressed without needing an additional brightness gate.

3. **Blob size filtering** — connected components smaller than `MIN_BLOB_AREA` px² are discarded. Wave noise and distant objects form many small scattered blobs (typically < 100 px²); real obstacles form larger connected regions that grow as the vessel closes in.

4. **Sector analysis** — the surviving obstacle pixels are counted in three horizontal sectors (left / centre / right). If the centre sector exceeds `DANGER_PX` pixels, `is_danger` is set.

5. **TTC proxy** — a time-to-collision estimate is computed from the mean frame-difference intensity in the forward corridor:
   ```
   TTC ≈ 255 / mean_pixel_change
   ```
   A faster approach (larger frame diff) produces a smaller TTC. Values below `CollisionAvoider.TTC_CRITICAL` (3 s) trigger an emergency stop.

### Tuning parameters

All parameters are class-level constants at the top of `ObstacleDetector`:

| Parameter | Default | Effect |
|---|---|---|
| `PROXIMITY_FRAC` | `0.44` | ROI start as fraction of frame height. Raise (e.g. `0.60`) to react only to closer objects; lower to detect earlier |
| `MIN_BLOB_AREA` | `200` | Minimum blob size in px². Raise (e.g. `400`) to suppress more wave noise; lower if small distant obstacles are missed |
| `DANGER_PX` | `200` | Centre-sector pixel count threshold for `is_danger`. Raise to reduce sensitivity; lower to react sooner |

These values were derived by analysing the MVTD 119-USV sequence directly:
- Obstacle blobs range from **292 px²** (far) to **1517 px²** (close)
- Wave noise blobs are consistently **< 100 px²**
- The obstacle first enters the frame at **~47% from the top**

---

## Collision avoidance logic

The image is divided into three horizontal sectors (left / centre / right). `CollisionAvoider` applies the following priority hierarchy each frame:

| Condition | Action |
|---|---|
| `TTC < 3 s` | Emergency stop |
| Centre sector clear | Pass through (no change to velocities) |
| All three sectors blocked | Full stop and wait |
| Centre blocked, one side clearer | Slow to 50 % speed, steer toward clearer side |

Obstacle observations are also written into the SLAM experience graph so the map records where hazards were seen — a foundation for future obstacle-aware path planning.

---

## Project structure

```
marine_spiking_slam/
├── .gitignore
├── pyproject.toml
└── src/
    ├── main.py               # Full pipeline entry point
    ├── maritime_slam.py      # Self-contained monolithic pipeline
    ├── eval.py               # Trajectory evaluation + plotting
    ├── visual_odometry.py
    ├── local_view.py
    ├── pose_cells.py
    ├── experience_map.py
    ├── obstacle_detection.py
    └── collision_avoidance.py
```

---

## References

- Milford, M. J., & Wyeth, G. F. (2008). *Mapping a suburb with a single camera using a biologically inspired SLAM system*. IEEE Transactions on Robotics.
- Ball, D., Heath, S., Wiles, J., Wyeth, G., Corke, P., & Milford, M. (2013). *OpenRatSLAM: an open source brain-based SLAM system*. Autonomous Robots.
- Farneback, G. (2003). *Two-frame motion estimation based on polynomial expansion*. SCIA.
