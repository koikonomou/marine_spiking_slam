import cv2
import numpy as np

class ObstacleDetector:
    def detect(self, frame):
        # 1. Setup ROI (Lower 50% of the image)
        h, w = frame.shape[:2]
        roi = frame[int(h*0.5):, :]
        
        # 2. Convert to Grayscale & Blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection (Using your confirmed values)
        edges = cv2.Canny(blurred, 85, 255)
        
        # 4. Brightness Thresholding (Using your confirmed value 200)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # 5. Combine (The "Secret Sauce" that made your test work)
        combined = cv2.bitwise_and(edges, thresh)
        
        # 6. Collision Logic: Check the middle "corridor" of the combined result
        # Corridor is the middle 33% of the width
        center_zone = combined[:, w//3 : 2*w//3]
        pixel_count = cv2.countNonZero(center_zone)
        
        # If more than 50 "combined" pixels are in front of us, it's a real dock
        is_danger = pixel_count > 50
        
        return combined, is_danger