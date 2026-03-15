import cv2
import numpy as np
import os
from pathlib import Path
from obstacle_detection import ObstacleDetector
from collision_avoidance import CollisionAvoider


class MaritimeSLAM:
    def __init__(self, width, height):
        # 1. Pose Cell Network Parameters
        self.pc_dim_xy = 21
        self.pc_dim_th = 36
        self.cells = np.zeros((self.pc_dim_th, self.pc_dim_xy, self.pc_dim_xy))
        self.cells[0, 10, 10] = 1.0  # Start in the centre

        # 2. Visual Templates (Local View Match)
        self.templates = []
        self.vt_threshold = 0.1

        # 3. Odometry State
        self.prev_gray = None
        self.fov_deg   = 60
        self.width     = width
        self.height    = height

        # 4. CV-based obstacle detection and avoidance (no distance sensors)
        self.detector = ObstacleDetector()
        self.avoider  = CollisionAvoider()

    def get_odometry(self, gray):
        """Calculates v_trans and v_rot based on non-circular image profile matching."""
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0

        curr_profile = np.mean(gray[int(self.height * 0.4):int(self.height * 0.6), :], axis=0)
        prev_profile = np.mean(self.prev_gray[int(self.height * 0.4):int(self.height * 0.6), :], axis=0)

        n = len(curr_profile)
        shifts = range(-20, 21)
        diffs = []
        for s in shifts:
            if s >= 0:
                diff = np.mean(np.abs(curr_profile[s:] - prev_profile[:n - s]))
            else:
                diff = np.mean(np.abs(curr_profile[:n + s] - prev_profile[-s:]))
            diffs.append(diff)

        best_shift = shifts[np.argmin(diffs)]
        v_rot   = (best_shift * self.fov_deg / self.width) * (np.pi / 180.0)
        v_trans = float(np.min(diffs)) * 10.0

        self.prev_gray = gray
        return v_trans, v_rot

    def run_pipeline(self, frame):
        """
        Run one frame through the full pipeline.

        Returns
        -------
        v_trans_cmd, v_rot_cmd : float  — avoidance-corrected velocity commands
        obstacle_mask          : uint8  — fused obstacle mask
        is_danger              : bool
        pixel_counts           : dict   {'left', 'center', 'right'}
        ttc_s                  : float  — time-to-collision estimate
        action                 : str    — avoidance action label
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Odometry
        v_trans, v_rot = self.get_odometry(gray)

        # 2. Pose cell path integration (theta only, simplified)
        shift_th = int(v_rot * self.pc_dim_th / (2 * np.pi))
        self.cells = np.roll(self.cells, shift_th, axis=0)

        # 3. Multi-modal CV obstacle detection
        obstacle_mask, is_danger, pixel_counts, ttc_s = self.detector.detect(frame)

        # 4. Collision avoidance steering
        v_trans_cmd, v_rot_cmd, action = self.avoider.avoid(
            pixel_counts, ttc_s, v_trans, v_rot
        )

        return v_trans_cmd, v_rot_cmd, obstacle_mask, is_danger, pixel_counts, ttc_s, action

# --- Execution Block ---
base_path = Path(os.path.expanduser("~/codes/datasets/Maritime_Visual_Tracking_Dataset_MVTD/train/119-USV/"))
frames = sorted([f.name for f in base_path.glob("*.jpg")])
# Filter for jpg files and ensure the list isn't empty
frames = sorted([f.name for f in base_path.glob("*.jpg")])

if not frames:
    print(f"Error: No images found in {base_path}")
    print("Check if the path is correct and contains .jpg files.")
    exit()

# Load initial image to get dimensions
first_image_path = str(base_path / frames[0])
sample_img = cv2.imread(first_image_path)

if sample_img is None:
    print(f"Error: Could not read the image file at {first_image_path}")
    exit()

h, w = sample_img.shape[:2]
slam = MaritimeSLAM(w, h)

for f_name in frames:
    frame_path = str(base_path / f_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    vt, vr, obs_mask, is_danger, pixel_counts, ttc_s, action = slam.run_pipeline(frame)

    display = CollisionAvoider.draw_hud(frame.copy(), pixel_counts, ttc_s, action)

    # Tint obstacle pixels red
    red_overlay = np.zeros_like(frame)
    red_overlay[obs_mask > 0] = [0, 0, 200]
    display = cv2.addWeighted(display, 1.0, red_overlay, 0.45, 0)

    cv2.imshow("Maritime SLAM & Obstacle Avoidance", display)
    cv2.imshow("Obstacle Mask", obs_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
