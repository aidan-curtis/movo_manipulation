from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui)
import numpy as np

class RandomSearch(Planner):
    def __init__(self):
        super(RandomSearch, self).__init__()
        self.step_size = [0.05, np.pi/18]

    def get_plan(self, environment):
        environment.setup()
        
        camera_pose, image_data = environment.get_robot_vision()
        environment.update_visibility(camera_pose, image_data)
        environment.update_occupancy(image_data)

        environment.plot_grids(visibility=False, occupancy=True)

        wait_if_gui()

