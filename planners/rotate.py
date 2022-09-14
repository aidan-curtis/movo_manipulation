from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import joint_from_name
from utils.graph import Graph
import numpy as np
import time

GRID_RESOLUTION = 0.2

class Rotate(Planner):
    def __init__(self, env):
        super(Rotate, self).__init__()
        self.env = env

    def get_plan(self, **kwargs):
        self.env.setup()
   
        self.base_joints = [
            joint_from_name(self.env.robot, "x"),
            joint_from_name(self.env.robot, "y"),
            joint_from_name(self.env.robot, "theta")]

        q = list(self.env.get_current_q(self.base_joints))
        for _ in range(32):
            q[2] = q[2]+0.2
            self.env.move_robot(q, self.base_joints) 
            time.sleep(0.1)
        print("Done")
