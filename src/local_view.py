import cv2
import numpy as np


class LocalViewMatcher:
    def __init__(self, template_x=60, template_y=10, match_threshold=0.1,
                 shift_match=5, step_match=1):
        self.template_x = template_x
        self.template_y = template_y
        self.match_threshold = match_threshold
        self.shift_match = shift_match
        self.step_match = step_match
        self.templates = []   # list of {'data': np.array, 'mean': float}
        self._epsilon = 0.005

    def get_view_template(self, frame):
        """Convert image to a normalized template. Returns (template, raw_mean)."""
        resized = cv2.resize(frame, (self.template_x, self.template_y))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        raw_mean = float(np.mean(gray))
        std = np.std(gray)
        if std > 0.01:
            gray = (gray - raw_mean) / std
        return gray.flatten(), raw_mean

    def compare(self, current_template, current_mean=None):
        """Compare current view to stored templates with fast rejection and shift matching."""
        if current_mean is None:
            current_mean = float(np.mean(current_template))

        if not self.templates:
            self.templates.append({'data': current_template, 'mean': current_mean})
            return 0, 1.0

        best_match_id = -1
        min_error = float('inf')

        for i, vt in enumerate(self.templates):
            # Fast rejection: skip if mean brightness differs too much
            if abs(current_mean - vt['mean']) > self.match_threshold + self._epsilon:
                continue

            # Shift matching to tolerate small horizontal displacements
            for shift in range(-self.shift_match, self.shift_match + 1, self.step_match):
                shifted = np.roll(current_template, shift)
                error = float(np.mean(np.abs(shifted - vt['data'])))
                if error < min_error:
                    min_error = error
                    best_match_id = i

        if min_error > self.match_threshold or best_match_id == -1:
            self.templates.append({'data': current_template, 'mean': current_mean})
            best_match_id = len(self.templates) - 1

        return best_match_id, min_error
