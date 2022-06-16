from environments.environment import Environment

from pybullet_planning.pybullet_tools.utils import (create_box, TAN, BROWN, 
                                                    set_pose, Pose, Point, LockRenderer,
                                                    set_joint_position, load_model)
import math
import random
import pybullet as p

class SingleMovable(Environment):
    def __init__(self):
        super(SingleMovable, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (6, 2, 0) # TODO: Create separate class for configuration space
        self.objects = []
        self.viewed_voxels = []

    def disconnect(self):
        try:
            p.disconnect()
        except:
            pass

    def setup(self):

        self.disconnect()
        
        p.connect(p.GUI)
        self.robot = self.setup_robot()

        with LockRenderer():


            blocking_box = create_box(3,4.5,1, mass=1, color=BROWN)
            set_pose(blocking_box,
                    Pose(point=Point(
                            x=3,
                            y=1.25,
                            z=1/2,
                        )
                    )
                )

            blocking_chair = load_model(
                    "../models/partnet_mobility/179/mobility.urdf", scale=0.5
                )

            set_joint_position(blocking_chair, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair,
                Pose(point=Point(
                        x=3,
                        y=4.25,
                        z=0.42,
                    )
                )
            )

            self.room = self.create_closed_room(length=6, width=10, center = [3,2], movable_obstacles=[blocking_chair])
            self.objects += [blocking_box, blocking_chair]
            self.static_objects = [blocking_box]
            self.centered_aabb = self.get_centered_aabb()
            self.setup_grids()