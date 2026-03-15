import cv2
import numpy as np


class CollisionAvoider:
    """
    Geometry-only collision avoidance for an unmanned surface vehicle.

    Uses the per-sector obstacle pixel counts and the time-to-collision
    estimate produced by ObstacleDetector to generate speed and steering
    commands — no distance sensors required.

    Decision hierarchy
    ------------------
    1. TTC below critical threshold → emergency stop (highest priority).
    2. Centre sector clear → pass-through (no change to commanded velocities).
    3. All three sectors blocked → full stop and wait.
    4. Centre blocked, one side clear → slow down and turn toward the
       less-blocked side.

    The avoider is stateless between calls so it can be dropped into any
    control loop without side effects.
    """

    DANGER_PX    = 50   # obstacle pixel threshold per sector
    TTC_CRITICAL = 3.0  # seconds — below this TTC triggers emergency stop
    TURN_GAIN    = 0.4  # radians added to v_rot per avoidance step
    SLOW_FACTOR  = 0.5  # speed multiplier while steering around an obstacle

    def avoid(self, pixel_counts, ttc_s, v_trans, v_rot):
        """
        Parameters
        ----------
        pixel_counts : dict  {'left', 'center', 'right'}  — from ObstacleDetector
        ttc_s        : float — time-to-collision estimate  (inf = no imminent threat)
        v_trans      : float — current forward speed command
        v_rot        : float — current rotation command (rad)

        Returns
        -------
        v_trans_cmd : float
        v_rot_cmd   : float
        action      : str — human-readable label for display / logging
        """
        center = pixel_counts['center']
        left   = pixel_counts['left']
        right  = pixel_counts['right']

        # Priority 1: imminent collision from optical flow TTC
        if ttc_s < self.TTC_CRITICAL:
            return 0.0, 0.0, 'EMERGENCY STOP'

        # Priority 2: forward corridor is clear
        if center < self.DANGER_PX:
            return v_trans, v_rot, 'CLEAR'

        # Priority 3: all sectors blocked — stop and wait
        if left > self.DANGER_PX and right > self.DANGER_PX:
            return 0.0, 0.0, 'ALL BLOCKED - STOP'

        # Priority 4: steer toward the less-obstructed side
        if left <= right:
            return v_trans * self.SLOW_FACTOR, v_rot - self.TURN_GAIN, 'AVOID LEFT'
        else:
            return v_trans * self.SLOW_FACTOR, v_rot + self.TURN_GAIN, 'AVOID RIGHT'

    @staticmethod
    def draw_hud(frame, pixel_counts, ttc_s, action):
        """
        Overlay a heads-up display on the frame showing:
          • Per-sector obstacle pixel counts with danger colouring
          • Sector divider lines
          • Time-to-collision estimate
          • Current avoidance action

        Returns the annotated frame (original is not modified).
        """
        h, w = frame.shape[:2]
        out  = frame.copy()

        sectors = [
            ('L', 0,       w // 3,     pixel_counts['left']),
            ('C', w // 3,  2 * w // 3, pixel_counts['center']),
            ('R', 2 * w // 3, w,       pixel_counts['right']),
        ]
        for label, x0, x1, count in sectors:
            danger = count > CollisionAvoider.DANGER_PX
            colour = (0, 0, 255) if danger else (0, 200, 0)
            x_mid  = (x0 + x1) // 2
            cv2.putText(out, f'{label}:{count}', (x_mid - 25, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
            if x1 < w:
                cv2.line(out, (x1, 0), (x1, h), (120, 120, 120), 1)

        # TTC readout
        ttc_str = f'TTC: {ttc_s:.1f}s' if ttc_s < 999 else 'TTC: --'
        cv2.putText(out, ttc_str, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Action label
        colour = (0, 255, 0) if action == 'CLEAR' else (0, 0, 255)
        cv2.putText(out, action, (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

        return out
