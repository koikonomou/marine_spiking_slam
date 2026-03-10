import cv2
import numpy as np
import os
from visual_odometry import VisualOdometry
from local_view import LocalViewMatcher
from experience_map import ExperienceMap
from obstacle_detection import ObstacleDetector
from pose_cells import PoseCells


dataset_path = "/home/kate/datasets/Maritime_Visual_Tracking_Dataset_MVTD/train/32-USV/"

def run_full_system():
    frames = sorted([f for f in os.listdir(dataset_path) if f.endswith('.jpg')])
    
    first_frame = cv2.imread(os.path.join(dataset_path, frames[0]))
    h, w = first_frame.shape[:2]
    
    vo = VisualOdometry(w, h)
    lv = LocalViewMatcher()
    pc = PoseCells()
    em = ExperienceMap()
    det = ObstacleDetector()

    traj_map = np.zeros((600, 600, 3), dtype=np.uint8)
    map_center = (300, 300)

    for frame_name in frames:
        img_path = os.path.join(dataset_path, frame_name)
        frame = cv2.imread(img_path)
        if frame is None: continue

        v_t, v_r = vo.calculate(frame)
        pc.integrate(v_t, v_r)
        
        prev_pos = (int(em.x * 10) + map_center[0], int(em.y * 10) + map_center[1])
        em.update(v_t, v_r)
        curr_pos = (int(em.x * 10) + map_center[0], int(em.y * 10) + map_center[1])
        
        cv2.line(traj_map, prev_pos, curr_pos, (0, 255, 255), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 85, 255)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_and(edges, thresh)
        collision_zone = combined[int(h*0.5):, int(w*0.33):int(w*0.66)]
        pixel_count = cv2.countNonZero(collision_zone)
        
        is_danger = pixel_count > 50 

        display = frame.copy()
        
        mask = combined > 0
        display[mask] = [0, 255, 0] 

        if is_danger:
            cv2.putText(display, "COLLISION ALERT", (w//4, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.rectangle(display, (0,0), (w-1, h-1), (0,0,255), 10)

        cv2.imshow('Camera Feed', display)
        cv2.imshow('Trajectory Map', traj_map)
        cv2.imshow('Obstacle Mask (Combined)', combined)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_full_system()