import numpy as np
from scipy.ndimage import gaussian_filter

class PoseCells:
    def __init__(self, dim_xy=21, dim_th=36):
        self.dim_xy = dim_xy
        self.dim_th = dim_th
        # 3D Grid: Theta, Y, X
        self.cells = np.zeros((dim_th, dim_xy, dim_xy))
        self.cells[0, dim_xy//2, dim_xy//2] = 1.0 

        self.vt_pose_map = {} 

    def integrate(self, v_trans, v_rot):

        shift_th = int(round(v_rot * self.dim_th / (2 * np.pi)))
        if shift_th != 0:
            self.cells = np.roll(self.cells, shift_th, axis=0)

        v_trans_scaled = v_trans * 0.5
        
        for th in range(self.dim_th):
            angle = th * (2 * np.pi / self.dim_th)
            dx = int(round(v_trans_scaled * np.cos(angle)))
            dy = int(round(v_trans_scaled * np.sin(angle)))
            
            if dx != 0 or dy != 0:
                self.cells[th, :, :] = np.roll(self.cells[th, :, :], shift=(dy, dx), axis=(0, 1))

        self._attractor_dynamics()

    def _attractor_dynamics(self):

        self.cells = gaussian_filter(self.cells, sigma=0.5, mode='wrap')
        inhibition = 0.0002
        self.cells = np.maximum(self.cells - inhibition, 0)
        

        total_energy = np.sum(self.cells)
        if total_energy > 0:
            self.cells /= total_energy

    def inject_energy(self, vt_id):
        best_pose = self.get_best_pose()
        
        if vt_id not in self.vt_pose_map:
            self.vt_pose_map[vt_id] = best_pose
        else:
            target_th, target_y, target_x = self.vt_pose_map[vt_id]
            energy_amt = 0.15 
            self.cells[target_th, target_y, target_x] += energy_amt
            self.cells /= np.sum(self.cells)

    def get_best_pose(self):
        idx = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        return idx