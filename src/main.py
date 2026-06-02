import cv2
import numpy as np
import os
from visual_odometry import VisualOdometry
from local_view import LocalViewMatcher
from experience_map import ExperienceMap
from obstacle_detection import ObstacleDetector
from collision_avoidance import CollisionAvoider
from pose_cells import PoseCells


dataset_path = os.path.expanduser("~/codes/datasets/Maritime_Visual_Tracking_Dataset_MVTD/train/119-USV/")


def run_full_system():
    frames = sorted([f for f in os.listdir(dataset_path) if f.endswith('.jpg')])

    first_frame = cv2.imread(os.path.join(dataset_path, frames[0]))
    h, w = first_frame.shape[:2]

    vo  = VisualOdometry(w, h)
    lv  = LocalViewMatcher()
    pc  = PoseCells()
    em  = ExperienceMap()
    det = ObstacleDetector()
    avo = CollisionAvoider()

    traj_map   = np.zeros((600, 600, 3), dtype=np.uint8)
    map_center = (300, 300)

    for frame_name in frames:
        img_path = os.path.join(dataset_path, frame_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # 1. Visual odometry
        v_t, v_r = vo.calculate(frame)

        # 2. Local view matching → energy injection into pose cells
        template, vt_mean = lv.get_view_template(frame)
        vt_id, vt_err     = lv.compare(template, vt_mean)
        pc.integrate(v_t, v_r)
        pc.inject_energy(vt_id)

        # 3. Experience map: accumulate odometry, create nodes, relax graph
        prev_pos = (int(em.x * 10) + map_center[0], int(em.y * 10) + map_center[1])
        em.update(v_t, v_r)
        em.relax()
        curr_pos = (int(em.x * 10) + map_center[0], int(em.y * 10) + map_center[1])
        cv2.line(traj_map, prev_pos, curr_pos, (0, 255, 255), 2)

        # 4. CV-based obstacle detection (water segmentation + optical flow + edges)
        obstacle_mask, is_danger, pixel_counts, ttc_s = det.detect(frame)

        # 5. Collision avoidance: compute corrected velocity commands
        v_t_cmd, v_r_cmd, action = avo.avoid(pixel_counts, ttc_s, v_t, v_r)

        # 6. Tag the current SLAM experience node with the danger flag so the
        #    map encodes where obstacles were observed during navigation.
        if em.experiences:
            em.experiences[em.current_exp_id]['danger'] = is_danger

        # 7. Trajectory map: mark dangerous nodes in red, safe in cyan
        node_colour = (0, 0, 255) if is_danger else (0, 255, 255)
        cv2.circle(traj_map, curr_pos, 3, node_colour, -1)

        # 8. Visualisation
        # Tint obstacle pixels red on the camera feed
        red_overlay = np.zeros_like(frame)
        red_overlay[obstacle_mask > 0] = [0, 0, 200]
        display = cv2.addWeighted(frame, 1.0, red_overlay, 0.45, 0)
        display = CollisionAvoider.draw_hud(display, pixel_counts, ttc_s, action)

        cv2.imshow('Camera Feed + Avoidance HUD', display)
        cv2.imshow('Trajectory Map (red = danger nodes)', traj_map)
        cv2.imshow('Obstacle Mask', obstacle_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_full_system()
