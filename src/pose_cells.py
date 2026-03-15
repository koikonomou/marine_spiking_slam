import numpy as np
from scipy.ndimage import gaussian_filter


class PoseCells:
    PC_CELL_X_SIZE = 1.0
    VT_INJECT_ENERGY = 0.15
    VT_ACTIVE_DECAY = 1.0
    VT_RESTORE = 0.05

    def __init__(self, dim_xy=21, dim_th=36):
        self.dim_xy = dim_xy
        self.dim_th = dim_th
        self.cells = np.zeros((dim_th, dim_xy, dim_xy))
        self.cells[dim_th // 2, dim_xy // 2, dim_xy // 2] = 1.0

        # vt_id -> dict with 'x','y','th','decay'
        self.vt_templates = {}

        # Best pose as floats (sub-cell, from population vector decoding)
        self.best_th = float(dim_th // 2)
        self.best_y = float(dim_xy // 2)
        self.best_x = float(dim_xy // 2)

    def integrate(self, v_trans, v_rot):
        """Path integration with bilinear XY interpolation and fractional theta shift."""
        v_trans = v_trans / self.PC_CELL_X_SIZE

        # Handle reverse motion
        angle_to_add = 0.0
        if v_trans < 0:
            v_trans = -v_trans
            angle_to_add = np.pi

        # Clamp so bilinear weights stay valid (weight_ne >= 0)
        v_trans = min(v_trans, 0.5)

        # XY bilinear interpolation per theta slice
        c_size_th = 2 * np.pi / self.dim_th
        for th_idx in range(self.dim_th):
            angle = th_idx * c_size_th + angle_to_add

            rot_quads = int(np.floor(angle * 2 / np.pi)) % 4
            dir90 = angle - np.floor(angle * 2 / np.pi) * np.pi / 2

            plane = np.rot90(self.cells[th_idx], rot_quads)

            v = v_trans
            wsw = v * v * np.cos(dir90) * np.sin(dir90)
            wse = v * np.sin(dir90) * (1.0 - v * np.cos(dir90))
            wnw = v * np.cos(dir90) * (1.0 - v * np.sin(dir90))
            wne = 1.0 - wsw - wse - wnw

            new_plane = (
                plane * wne
                + np.roll(plane, 1, axis=1) * wse
                + np.roll(plane, 1, axis=0) * wnw
                + np.roll(np.roll(plane, 1, axis=1), 1, axis=0) * wsw
            )
            self.cells[th_idx] = np.rot90(new_plane, (4 - rot_quads) % 4)

        # Fractional theta rotation
        if v_rot != 0:
            weight = (abs(v_rot) / c_size_th) % 1.0
            if weight == 0:
                weight = 1.0
            sign = 1 if v_rot > 0 else -1
            shift1 = int(sign * np.floor(abs(v_rot) / c_size_th))
            shift2 = int(sign * np.ceil(abs(v_rot) / c_size_th))
            old = self.cells.copy()
            self.cells = (
                np.roll(old, shift1, axis=0) * (1.0 - weight)
                + np.roll(old, shift2, axis=0) * weight
            )

        self._attractor_dynamics()
        self._find_best()

    def _attractor_dynamics(self):
        self.cells = gaussian_filter(self.cells, sigma=0.5, mode='wrap')
        self.cells = np.maximum(self.cells - 0.00002, 0)
        total = np.sum(self.cells)
        if total > 0:
            self.cells /= total

    def _find_best(self):
        """Population vector decoding for sub-cell accuracy (mirrors RatSLAM find_best)."""
        max_idx = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        max_th, max_y, max_x = int(max_idx[0]), int(max_idx[1]), int(max_idx[2])

        cells_to_avg = 3
        sin_x = cos_x = sin_y = cos_y = sin_th = cos_th = 0.0

        for dth in range(2 * cells_to_avg + 1):
            for dy in range(2 * cells_to_avg + 1):
                for dx in range(2 * cells_to_avg + 1):
                    th = (max_th - cells_to_avg + dth) % self.dim_th
                    y  = (max_y  - cells_to_avg + dy)  % self.dim_xy
                    x  = (max_x  - cells_to_avg + dx)  % self.dim_xy
                    w = self.cells[th, y, x]
                    sin_x  += np.sin((x  + 1) * 2 * np.pi / self.dim_xy) * w
                    cos_x  += np.cos((x  + 1) * 2 * np.pi / self.dim_xy) * w
                    sin_y  += np.sin((y  + 1) * 2 * np.pi / self.dim_xy) * w
                    cos_y  += np.cos((y  + 1) * 2 * np.pi / self.dim_xy) * w
                    sin_th += np.sin((th + 1) * 2 * np.pi / self.dim_th) * w
                    cos_th += np.cos((th + 1) * 2 * np.pi / self.dim_th) * w

        self.best_x  = (np.arctan2(sin_x,  cos_x)  * self.dim_xy / (2 * np.pi) - 1.0) % self.dim_xy
        self.best_y  = (np.arctan2(sin_y,  cos_y)  * self.dim_xy / (2 * np.pi) - 1.0) % self.dim_xy
        self.best_th = (np.arctan2(sin_th, cos_th) * self.dim_th / (2 * np.pi) - 1.0) % self.dim_th

    def inject_energy(self, vt_id):
        """Inject energy at the stored pose for a visual template, with decay."""
        if vt_id not in self.vt_templates:
            self.vt_templates[vt_id] = {
                'x': self.best_x, 'y': self.best_y, 'th': self.best_th,
                'decay': self.VT_ACTIVE_DECAY,
            }
            return

        vt = self.vt_templates[vt_id]
        vt['decay'] += self.VT_ACTIVE_DECAY

        # Only inject energy for templates that are old enough (avoids self-reinforcement)
        if vt_id + 10 < len(self.vt_templates):
            energy = self.VT_INJECT_ENERGY * (1.0 / 30.0) * (30.0 - np.exp(1.2 * vt['decay']))
            if energy > 0:
                th = int(vt['th']) % self.dim_th
                y  = int(vt['y'])  % self.dim_xy
                x  = int(vt['x'])  % self.dim_xy
                self.cells[th, y, x] += energy
                total = np.sum(self.cells)
                if total > 0:
                    self.cells /= total

        # Restore decay for all templates
        for v in self.vt_templates.values():
            v['decay'] -= self.VT_RESTORE
            if v['decay'] < self.VT_ACTIVE_DECAY:
                v['decay'] = self.VT_ACTIVE_DECAY

    def get_best_pose(self):
        """Returns (th, y, x) as floats from population vector decoding."""
        return self.best_th, self.best_y, self.best_x
