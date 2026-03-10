import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from visual_odometry import VisualOdometry
from local_view import LocalViewMatcher
from experience_map import ExperienceMap

def load_ground_truth(path):
    # MVTD groundtruth is usually space or comma separated
    try:
        return np.loadtxt(path, delimiter=None) # Adjust delimiter if needed
    except:
        return None

def run_full_system():
    # Paths
    VIDEO_DIR = Path("/home/kate/datasets/Maritime_Visual_Tracking_Dataset_MVTD/train/119-USV//").expanduser()
    GT_PATH = VIDEO_DIR / "groundtruth.txt"
    
    frames = sorted(list(VIDEO_DIR.glob("*.jpg")))
    gt_data = load_ground_truth(GT_PATH)
    
    # Init modules
    sample = cv2.imread(str(frames[0]))
    vo = VisualOdometry(sample.shape[1], sample.shape[0])
    lv = LocalViewMatcher()
    em = ExperienceMap()

    print("Navigating Harbor...")
    for i, p in enumerate(frames):
        img = cv2.imread(str(p))
        
        # SLAM Pipeline
        vt, vr = vo.calculate(img)
        curr_template = lv.get_view_template(img)
        lv.compare(curr_template)
        em.add_experience(vt, vr)

        # Optional: Show video
        cv2.imshow("Processing", img)
        if cv2.waitKey(1) == ord('q'): break

    cv2.destroyAllWindows()

    # --- FINAL PLOT ---
    slam_path = np.array(em.path_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(slam_path[:, 0], slam_path[:, 1], label='RatSLAM Path', color='blue', linewidth=2)
    
    if gt_data is not None:
        # Note: We take the center of the bounding box as the (x,y) if GT is [x,y,w,h]
        gt_x = gt_data[:, 0] + gt_data[:, 2]/2
        gt_y = gt_data[:, 1] + gt_data[:, 3]/2
        plt.plot(gt_x[:len(slam_path)], gt_y[:len(slam_path)], 
                 label='Ground Truth', color='red', linestyle='--')

    plt.title("Maritime Navigation: SLAM vs Ground Truth")
    plt.xlabel("X Position (meters approx)")
    plt.ylabel("Y Position (meters approx)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_full_system()