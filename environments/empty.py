from environments.environment import Environment

from pybullet_planning.pybullet_tools.utils import create_box, TAN
import pybullet as p

def create_pillar(width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
    return  create_box(w=width, l=length, h=height, color=color, **kwargs)
    

class Empty(Environment):
    def __init__(self):
        super(Empty, self).__init__()

        self.setup()
        self.start = [0, 0]
        self.goal = [0, 2] # TODO: Create separate class for configuration space

    def setup(self):
        p.connect(p.DIRECT)

        floor_size = 6
        self.floor = create_pillar(width=floor_size, length=floor_size, color=TAN)
        self.robot = self.setup_robot()