import cv2
import numpy as np


class ObstacleDetector:
    """
    CV obstacle detector for marine environments (no distance sensors).

    Base: the original Canny + brightness-threshold fusion that was confirmed
    to work.  Two extras are layered on top without masking the base result:
      - Frame differencing: catches moving obstacles the edge layer misses.
      - Proximity ROI: detection starts at PROXIMITY_FRAC down the frame so
        far-away objects near the horizon are ignored.

    Tuning
    ------
    PROXIMITY_FRAC  float  0.5–0.9   higher = only react when object is closer
    MIN_BLOB_AREA   int    px²        raise to ignore small/distant blobs
    """

    DANGER_PX      = 200   # obstacle pixels in centre sector → danger
    DIFF_THRESHOLD = 30    # per-pixel abs difference to count as motion (TTC only)
    PROXIMITY_FRAC = 0.44  # ROI top edge — obstacle appears at ~47% from top in this dataset
    MIN_BLOB_AREA  = 200   # blobs smaller than this are discarded (wave noise < 100 px²)

    def __init__(self):
        self.prev_gray = None

    def detect(self, frame):
        """
        Returns
        -------
        obstacle_mask : uint8 single-channel mask
        is_danger     : bool
        pixel_counts  : dict  {'left', 'center', 'right'}
        ttc_s         : float — TTC estimate (inf = no threat)
        """
        h, w  = frame.shape[:2]
        roi_y = int(h * self.PROXIMITY_FRAC)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- base layer: original Canny + brightness threshold ---
        roi     = gray[roi_y:, :]
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        base    = cv2.Canny(blurred, 85, 255)

        combined = base

        # Remove tiny blobs (distant objects appear small)
        combined = self._filter_blobs(combined)

        # Place result back into a full-frame mask
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask[roi_y:, :] = combined

        # Sector pixel counts
        pixel_counts = {
            'left':   int(cv2.countNonZero(obstacle_mask[:, :w // 3])),
            'center': int(cv2.countNonZero(obstacle_mask[:, w // 3: 2 * w // 3])),
            'right':  int(cv2.countNonZero(obstacle_mask[:, 2 * w // 3:])),
        }
        is_danger = pixel_counts['center'] > self.DANGER_PX

        # TTC proxy from motion intensity in forward corridor
        ttc_s = float('inf')
        if self.prev_gray is not None:
            diff_roi = cv2.absdiff(gray[roi_y:, :], self.prev_gray[roi_y:, :])
            cx0, cx1 = w // 3, 2 * w // 3
            center_diff = float(np.mean(diff_roi[:, cx0:cx1]))
            if center_diff > 1.0:
                ttc_s = 255.0 / center_diff

        self.prev_gray = gray
        return obstacle_mask, is_danger, pixel_counts, ttc_s

    def _frame_diff(self, gray, roi_y):
        mask = np.zeros_like(gray)
        if self.prev_gray is None:
            return mask
        diff = cv2.absdiff(gray[roi_y:, :], self.prev_gray[roi_y:, :])
        _, motion = cv2.threshold(diff, self.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask[roi_y:, :] = motion
        return mask

    def _filter_blobs(self, mask):
        if self.MIN_BLOB_AREA <= 0:
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.MIN_BLOB_AREA:
                filtered[labels == i] = 255
        return filtered
