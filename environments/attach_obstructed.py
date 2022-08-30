from environments.environment import Environment, Room, GRID_HEIGHT, LIGHT_GREY
from pybullet_planning.pybullet_tools.utils import (set_pose, set_joint_position, Pose, Point,
                                                    load_model, create_box, TAN, BROWN,
                                                    LockRenderer, AABB, get_aabb)
import random
import math
import pybullet as p
import numpy as np
   

class AttObs(Environment):
    def __init__(self):
        super(AttObs, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (4, 0, round(np.pi, 3)) # TODO: Create separate class for configuration space

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
        
        p.connect(p.GUI)
        self.robot = self.setup_robot()

        with LockRenderer():

            blocking_chair = load_model(
                    "../models/partnet_mobility/179/mobility.urdf", scale=0.5
                )
            blocking_box = create_box(1, 1.5, 1, mass=1, color=BROWN)
            self.room = self.create_room(movable_obstacles=[blocking_chair, blocking_box])
            set_joint_position(blocking_chair, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair,
                Pose(point=Point(
                        x=2,
                        y=1.2,
                        z=0.42,
                    )
                )
            )

            set_pose(blocking_box,
                     Pose(point=Point(x=2+0.1, y=0, z=1 / 2), euler=[0, 0, np.pi/2]))
            chair_aabb = get_aabb(blocking_chair)
            self.objects_prop[blocking_chair] = [chair_aabb.upper[0] - chair_aabb.lower[0],
                                                 chair_aabb.upper[1] - chair_aabb.lower[1],
                                                 chair_aabb.upper[2] - chair_aabb.lower[2],
                                                 1]


            self.objects_prop[blocking_box] = [2, 4.5, 1, 1]

            self.objects += [blocking_box, blocking_chair]
            self.push_only = [blocking_box]
            self.static_objects = []
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()
            self.setup_grids()


    def create_room(self, movable_obstacles=[]):
        width = 4
        length = 6
        wall_height = 2
        center = [2,0]

        hall_length = 3
        floor1 = self.create_pillar(width=width+0.1, length=length-0.2, color=TAN)
        set_pose(floor1, Pose(Point(x=1+0.05, y=0)))

        floor2 = self.create_pillar(width=1.9, length=1.9, color=TAN)
        set_pose(floor2, Pose(Point(x=4+0.05, y=0)))


        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width+0.1, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1, Pose(point=Point(x=center[0]+0.05-1, y=center[1]+length/2+wall_thickness/2-0.21+0.1, z=wall_height/2)))

        wall_2 = self.create_pillar(width=width+0.1, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2, Pose(point=Point(x=center[0]+0.05-1, y=center[1]-(length/2+wall_thickness/2)+0.1, z=wall_height/2)))

        wall_3 = self.create_pillar(length=length-0.1-wall_thickness, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3, Pose(point=Point(y=0, x=center[0]-(width/2+wall_thickness/2)-1, z=wall_height/2)))

        wall_4 = self.create_pillar(length=2-0.05, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4, Pose(
            point=Point(y=2-0.1+0.025, x=3+0.1+wall_thickness/2, z=wall_height / 2)))

        wall_5 = self.create_pillar(length=2-0.05, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5, Pose(
            point=Point(y=-(2-0.1+0.025), x=3 + 0.1 + wall_thickness / 2, z=wall_height / 2)))

        wall_6 = self.create_pillar(length=wall_thickness, width=2-0.1, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_6, Pose(
            point=Point(y=1, x=4+0.05, z=wall_height / 2)))

        wall_7 = self.create_pillar(length=wall_thickness, width=2 - 0.1, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_7, Pose(
            point=Point(y=-1, x=4 + 0.05, z=wall_height / 2)))

        wall_8 = self.create_pillar(length=2, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_8, Pose(
            point=Point(y=0, x=5+wall_thickness/2, z=wall_height / 2)))


        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6, wall_7, wall_8]
        floors = [floor1, floor2]
        aabb = AABB(lower=(-1, -3, 0.05),
                    upper=(5, 3, 0 + GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room
