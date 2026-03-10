import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.prev_profile = None
        self.fov_rad = np.deg2rad(60)

    def get_profile(self, frame):
        # Focus on the horizon (middle of the image)
        roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = roi[int(self.height*0.4):int(self.height*0.6), :]
        return np.mean(roi, axis=0) / 255.0

    def calculate(self, frame):
        curr_profile = self.get_profile(frame)
        if self.prev_profile is None:
            self.prev_profile = curr_profile
            return 0.0, 0.0

        # Match profiles
        diffs = []
        shifts = range(-30, 31)
        for s in shifts:
            shifted = np.roll(curr_profile, s)
            diffs.append(np.mean(np.abs(shifted - self.prev_profile)))
        
        best_shift = shifts[np.argmin(diffs)]
        v_rot = (best_shift / self.width) * self.fov_rad
        v_trans = np.min(diffs) * 5.0 # Empirical scaling
        
        self.prev_profile = curr_profile
        return v_trans, v_rot