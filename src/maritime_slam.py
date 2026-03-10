import cv2
import numpy as np
import os
from pathlib import Path

class MaritimeSLAM:
    def __init__(self, width, height):
        # 1. Pose Cell Network Parameters
        self.pc_dim_xy = 21
        self.pc_dim_th = 36
        self.cells = np.zeros((self.pc_dim_th, self.pc_dim_xy, self.pc_dim_xy))
        self.cells[0, 10, 10] = 1.0  # Start in the center
        
        # 2. Visual Templates (Local View Match)
        self.templates = []
        self.vt_threshold = 0.1
        
        # 3. Odometry State
        self.prev_gray = None
        self.fov_deg = 60
        self.width = width
        self.height = height

    def get_odometry(self, gray):
        """Calculates v_trans and v_rot based on image differencing."""
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0
        
        # Collapse into 1D profile (LocalViewMatch logic)
        curr_profile = np.mean(gray[int(self.height*0.4):int(self.height*0.6), :], axis=0)
        prev_profile = np.mean(self.prev_gray[int(self.height*0.4):int(self.height*0.6), :], axis=0)
        
        # Find horizontal shift for v_rot
        diffs = []
        shifts = range(-20, 21)
        for s in shifts:
            shifted = np.roll(curr_profile, s)
            diffs.append(np.mean(np.abs(shifted - prev_profile)))
            
        best_shift = shifts[np.argmin(diffs)]
        v_rot = (best_shift * self.fov_deg / self.width) * (np.pi / 180.0)
        v_trans = np.min(diffs) * 10.0 # Scaling factor
        
        self.prev_gray = gray
        return v_trans, v_rot

    def detect_obstacles(self, frame):
        """Canny and Thresholding for Docks."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Mask the top 40% (Sky)
        edges[:int(self.height*0.4), :] = 0
        return edges

    def run_pipeline(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Get Motion
        v_trans, v_rot = self.get_odometry(gray)
        
        # 2. Update Pose Cells (Simplified Path Integration)
        # Shift cells based on v_rot (Theta axis)
        shift_th = int(v_rot * self.pc_dim_th / (2 * np.pi))
        self.cells = np.roll(self.cells, shift_th, axis=0)
        
        # 3. Detect Obstacles
        obstacles = self.detect_obstacles(frame)
        
        return v_trans, v_rot, obstacles

# --- Execution Block ---
base_path = os.path.expanduser("~/datasets/Maritime_Visual_Tracking_Dataset_MVTD/train/119-USV/")
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
    frame = cv2.imread(os.path.join(dataset_path, f_name))
    vt, vr, obs = slam.run_pipeline(frame)
    
    # Visualization
    display = frame.copy()
    # Draw "Danger" if many edges detected in front
    if np.sum(obs[:, w//3 : 2*w//3]) > 5000:
        cv2.putText(display, "OBSTACLE DETECTED", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Maritime SLAM & Obstacle Avoidance", display)
    cv2.imshow("Canny Filter", obs)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
