from planners.planner import Planner
from utils.utils import get_pointcloud_from_camera_image
from pybullet_planning.pybullet_tools.utils import wait_if_gui

class RandomSearch(Planner):
    def __init__(self):
        super(RandomSearch, self).__init__()

    def get_plan(self, environment):
        environment.setup()
        
        camera_pose, image_data = environment.get_robot_vision()
        environment.update_visibility(camera_pose, image_data)
        environment.update_occupancy(image_data)

        environment.plot_grids(visibility=True, occupancy=False)

        # get_pointcloud_from_camera_image(image_data)

        wait_if_gui()