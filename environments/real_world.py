from environments.environment import Environment, Room, LIGHT_GREY, GRID_HEIGHT
from pybullet_planning.pybullet_tools.utils import (TAN, AABB, set_pose, Pose, Point, LockRenderer)
import pybullet as p
import numpy as np


class RealWorld(Environment):
    def __init__(self):
        super(RealWorld, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (0, 0, 1)  # TODO: Create separate class for configuration space
        self.objects = []
        self.viewed_voxels = []

        # Properties represented as a list of width, length, height, mass
        self.objects_prop = dict()

    def setup(self):

        self.disconnect()
        self.connect()

        with LockRenderer():
            self.display_goal(self.goal)
            self.robot = self.setup_robot()
            self.room = self.create_room()
            self.static_objects = []
            self.setup_grids()
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()

    def create_room(self, **kwargs):
        width = 8
        floor1 = self.create_pillar(width=width, length=width, color=TAN)
        set_pose(floor1, Pose(Point(x=0, y=0)))

        aabb = AABB(lower=(-width/2.0, -width/2.0, 0.05),
                    upper=(6, 6, GRID_HEIGHT))
        room = Room([], [], aabb, [])
        return room
