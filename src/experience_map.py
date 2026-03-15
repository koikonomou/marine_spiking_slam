import math


class ExperienceMap:
    EXP_CORRECTION = 0.5  # spring correction factor (matches RatSLAM default)
    EXP_LOOPS = 10        # relaxation iterations per call

    def __init__(self):
        self.experiences = []   # list of dicts: {id, x, y, th, links_from, links_to}
        self.links = []         # list of dicts: {exp_from_id, exp_to_id, d, heading_rad, facing_rad}
        self.current_exp_id = 0
        self.prev_exp_id = 0

        # Accumulated odometry since last experience node
        self.accum_x = 0.0
        self.accum_y = 0.0
        self.accum_th = 0.0

        # Current best-estimate pose (kept in sync with odometry)
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        # Path history for plotting / evaluation
        self.path_history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clip_rad(self, r):
        while r > math.pi:  r -= 2 * math.pi
        while r < -math.pi: r += 2 * math.pi
        return r

    def _signed_delta_rad(self, r1, r2):
        d = r2 - r1
        while d > math.pi:  d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    # ------------------------------------------------------------------
    # Core interface (mirrors RatSLAM on_odo / on_create_experience etc.)
    # ------------------------------------------------------------------

    def on_odo(self, vtrans, vrot, time_diff=1.0):
        """Accumulate odometry since the last experience node."""
        vtrans *= time_diff
        vrot   *= time_diff
        self.accum_th = self._clip_rad(self.accum_th + vrot)
        self.accum_x += vtrans * math.cos(self.accum_th)
        self.accum_y += vtrans * math.sin(self.accum_th)

        if self.experiences:
            exp = self.experiences[self.current_exp_id]
            self.x  = exp['x'] + self.accum_x
            self.y  = exp['y'] + self.accum_y
            self.th = self.accum_th
        else:
            self.x  = self.accum_x
            self.y  = self.accum_y
            self.th = self.accum_th

        self.path_history.append((self.x, self.y))

    def create_experience(self):
        """Create a new experience node at the current estimated pose."""
        if self.experiences:
            parent = self.experiences[self.current_exp_id]
            new_x  = parent['x'] + self.accum_x
            new_y  = parent['y'] + self.accum_y
        else:
            new_x, new_y = self.accum_x, self.accum_y

        exp = {
            'id': len(self.experiences),
            'x': new_x,
            'y': new_y,
            'th': self._clip_rad(self.accum_th),
            'links_from': [],
            'links_to':   [],
        }
        self.experiences.append(exp)
        new_id = len(self.experiences) - 1

        if new_id > 0:
            self._create_link(self.current_exp_id, new_id)

        self.prev_exp_id    = self.current_exp_id
        self.current_exp_id = new_id
        self._reset_accumulator()
        return new_id

    def set_experience(self, exp_id, rel_rad=0.0):
        """Switch to an existing experience (called on loop closure)."""
        if exp_id >= len(self.experiences) or exp_id == self.current_exp_id:
            return
        self._create_link(self.current_exp_id, exp_id)
        self.prev_exp_id    = self.current_exp_id
        self.current_exp_id = exp_id
        exp = self.experiences[exp_id]
        self.accum_th = self._clip_rad(exp['th'] + rel_rad)
        self.accum_x  = 0.0
        self.accum_y  = 0.0

    def relax(self):
        """Spring-based graph relaxation for loop-closure correction (RatSLAM iterate())."""
        for _ in range(self.EXP_LOOPS):
            for exp in self.experiences:
                for li in exp['links_from']:
                    link    = self.links[li]
                    exp_to  = self.experiences[link['exp_to_id']]

                    # Where exp_from expects exp_to to be
                    lx = exp['x'] + link['d'] * math.cos(exp['th'] + link['heading_rad'])
                    ly = exp['y'] + link['d'] * math.sin(exp['th'] + link['heading_rad'])

                    # Correct both nodes symmetrically
                    dx = (exp_to['x'] - lx) * self.EXP_CORRECTION
                    dy = (exp_to['y'] - ly) * self.EXP_CORRECTION
                    exp['x']    += dx;  exp['y']    += dy
                    exp_to['x'] -= dx;  exp_to['y'] -= dy

                    # Correct facing angles
                    df = self._signed_delta_rad(exp['th'] + link['facing_rad'], exp_to['th'])
                    exp['th']    = self._clip_rad(exp['th']    + df * self.EXP_CORRECTION)
                    exp_to['th'] = self._clip_rad(exp_to['th'] - df * self.EXP_CORRECTION)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_link(self, from_id, to_id):
        exp_from = self.experiences[from_id]
        # Deduplicate
        for li in exp_from['links_from']:
            if self.links[li]['exp_to_id'] == to_id:
                return

        d       = math.sqrt(self.accum_x ** 2 + self.accum_y ** 2)
        heading = self._signed_delta_rad(exp_from['th'], math.atan2(self.accum_y, self.accum_x))
        facing  = self._signed_delta_rad(exp_from['th'], self._clip_rad(self.accum_th))

        link = {
            'exp_from_id': from_id,
            'exp_to_id':   to_id,
            'd':           d,
            'heading_rad': heading,
            'facing_rad':  facing,
        }
        link_id = len(self.links)
        self.links.append(link)
        self.experiences[from_id]['links_from'].append(link_id)
        self.experiences[to_id  ]['links_to'  ].append(link_id)

    def _reset_accumulator(self):
        exp = self.experiences[self.current_exp_id]
        self.accum_th = self._clip_rad(exp['th'])
        self.accum_x  = 0.0
        self.accum_y  = 0.0

    # ------------------------------------------------------------------
    # Compatibility shims (used by main.py and eval.py)
    # ------------------------------------------------------------------

    def update(self, v_trans, v_rot):
        """Backward-compatible: accumulate odometry and auto-create experience nodes."""
        self.on_odo(v_trans, v_rot)
        dist = math.sqrt(self.accum_x ** 2 + self.accum_y ** 2)
        if not self.experiences or dist > 1.0:
            self.create_experience()
        return {'x': self.x, 'y': self.y}

    def add_experience(self, v_trans, v_rot):
        """Alias used by eval.py."""
        return self.update(v_trans, v_rot)
