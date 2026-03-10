import math

class ExperienceMap:
    def __init__(self):
        self.nodes = [] 
        self.x, self.y, self.th = 0.0, 0.0, 0.0
        self.path_history = [] 

    def update(self, v_trans, v_rot):
        self.th += v_rot
        self.x += v_trans * math.cos(self.th)
        self.y += v_trans * math.sin(self.th)
        
        self.nodes.append({'x': self.x, 'y': self.y})
        self.path_history.append((self.x, self.y))
        return self.nodes[-1]