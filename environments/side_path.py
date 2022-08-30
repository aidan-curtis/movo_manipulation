#from environments.environment import Environment
from environments.environment import Environment

from pybullet_planning.pybullet_tools.utils import (create_box, BROWN, 
                                                    set_pose, Pose, Point, LockRenderer)
import pybullet as p

class SidePath(Environment):
    def __init__(self):
        super(SidePath, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (6, 2, 0) # TODO: Create separate class for configuration space
        self.objects = []
        self.viewed_voxels = []

        # Properties represented as a list of width, length, height, mass
        self.objects_prop = dict()

    def disconnect(self):
        try:
            p.disconnect()
        except:
            pass

    def setup(self):

        self.disconnect()
        self.connect()

        self.robot = self.setup_robot()

        with LockRenderer():
            self.room = self.create_closed_room(length=6, width=10, center = [3,2])


            blocking_box = create_box(3,4,1, mass=1, color=BROWN)
            set_pose(blocking_box,
                    Pose(point=Point(
                            x=3,
                            y=1,
                            z=1/2,
                        )
                    )
                )
            self.objects_prop[blocking_box] = [3, 4, 1, 1]

            self.setup_grids()
        self.centered_aabb = self.get_centered_aabb()
        self.centered_oobb = self.get_centered_oobb()
        self.objects += [blocking_box]
        self.static_objects = [blocking_box]