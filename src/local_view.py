import cv2
import numpy as np

class LocalViewMatcher:
    def __init__(self, template_x=60, template_y=10, match_threshold=0.1):
        self.template_x = template_x
        self.template_y = template_y
        self.match_threshold = match_threshold
        self.templates = [] # List of stored Visual Templates

    def get_view_template(self, frame):
        """Converts image to a normalized template (convert_view_to_view_template)"""
        # 1. Resize to a small template size for fast matching
        resized = cv2.resize(frame, (self.template_x, self.template_y))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        
        # 2. Patch Normalization
        mean = np.mean(gray)
        std = np.std(gray)
        if std > 0.01:
            gray = (gray - mean) / std
        
        return gray.flatten()

    def compare(self, current_template):
        #Compares current view to all stored templates
        if not self.templates:
            self.templates.append(current_template)
            return 0, 1.0 # First VT

        best_match_id = -1
        min_error = float('inf')

        for i, vt in enumerate(self.templates):
            # Mean Absolute Error comparison
            error = np.mean(np.abs(current_template - vt))
            if error < min_error:
                min_error = error
                best_match_id = i

        # If error is too high, it's a new place
        if min_error > self.match_threshold:
            self.templates.append(current_template)
            best_match_id = len(self.templates) - 1

        return best_match_id, min_error