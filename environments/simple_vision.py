from environments.vamp_environment import Environment, Room, LIGHT_GREY, GRID_HEIGHT

from pybullet_planning.pybullet_tools.utils import (TAN, AABB, set_pose, Pose, Point, LockRenderer)
import pybullet as p
import numpy as np


class SimpleVision(Environment):
    def __init__(self):
        super(SimpleVision, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (2, 0, 0)  # TODO: Create separate class for configuration space
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

            self.room = self.create_room()
            self.static_objects = []
            self.setup_grids()
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()

    def create_room(self, movable_obstacles=[]):
        width = 6
        length = 6
        wall_height = 2
        center = [2, 2]


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

        wall_5 = self.create_pillar(length=4.7, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5,
                 Pose(point=Point(y=1.35, x=1.1, z=wall_height / 2)))

        wall_6 = self.create_pillar(length=wall_thickness, width=2.5, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_6,
                 Pose(point=Point(y=1.5, x=2.4, z=wall_height / 2)))

        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6]
        floors = [floor1]
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room
