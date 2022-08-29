#from environments.environment import Environment, Room, LIGHT_GREY, GRID_HEIGHT
from environments.vamp_environment import Environment, Room, LIGHT_GREY, GRID_HEIGHT

from pybullet_planning.pybullet_tools.utils import (create_box, TAN, BROWN, AABB,
                                                    set_pose, Pose, Point, LockRenderer,
                                                    set_joint_position, load_model, get_aabb)
import math
import random
import pybullet as p
import numpy as np


class SingleHallway(Environment):
    def __init__(self):
        super(SingleHallway, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (2.4, 0, round(np.pi/2,3))  # TODO: Create separate class for configuration space
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

            self.room = self.create_room()
            self.static_objects = []
            self.setup_grids()
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()

    def create_room(self, movable_obstacles=[]):
        width = 4
        length = 6
        wall_height = 2
        center = [1, 2]


        floor1 = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor1, Pose(Point(x=center[0], y=center[1])))

        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1,
                 Pose(point=Point(x=center[0], y=center[1] + length / 2 + wall_thickness / 2, z=wall_height / 2)))

        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2,
                 Pose(point=Point(x=center[0], y=center[1] - (length / 2 + wall_thickness / 2), z=wall_height / 2)))

        wall_3 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3,
                 Pose(point=Point(y=center[1], x=center[0] + width / 2 + wall_thickness / 2, z=wall_height / 2)))

        wall_4 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4,
                 Pose(point=Point(y=center[1], x=center[0] - (width / 2 + wall_thickness / 2), z=wall_height / 2)))

        wall_5 = self.create_pillar(length=3.9, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5,
                 Pose(point=Point(y=0.95, x=1.7, z=wall_height / 2)))

        walls = [wall_1, wall_2, wall_3, wall_4, wall_5]
        floors = [floor1]
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room
