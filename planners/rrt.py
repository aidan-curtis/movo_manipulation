from planners.planner import Planner
from utils.utils import get_pointcloud_from_camera_image
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions)
import numpy as np
from itertools import product
import time


class RRT(Planner):
    def __init__(self):
        super(RRT, self).__init__()
        self.step_size = [0.05, np.pi/18]

    def get_plan(self, environment):
        environment.setup()
        
        camera_pose, image_data = environment.get_robot_vision()
        environment.update_visibility(camera_pose, image_data)
        environment.update_occupancy(image_data)

        environment.plot_grids(visibility=False, occupancy=True)

        wait_if_gui()

