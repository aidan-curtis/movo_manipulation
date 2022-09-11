from environments.environment import Environment, Room, LIGHT_GREY, GRID_HEIGHT

from pybullet_planning.pybullet_tools.utils import (create_box, TAN, BROWN, AABB,
                                                    set_pose, Pose, Point, LockRenderer,
                                                    set_joint_position, get_aabb, joint_from_name,
                                                    set_joint_positions)
import math
import random
import pybullet as p
import numpy as np


class SimpleNamo(Environment):
    def __init__(self, **kwargs):
        super(SimpleNamo, self).__init__(**kwargs)

        self.start = (0, 0, 0)
        self.goal = (4, 0, 0)  # TODO: Create separate class for configuration space
        self.objects = []
        self.viewed_voxels = []

        # Properties represented as a list of width, length, height, mass
        self.objects_prop = dict()

    def setup(self):

        self.disconnect()
        self.connect()

        with LockRenderer():
            self.display_goal(self.goal)
            # These 3 lines are important and should be located here
            self.robot = self.setup_robot()
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()

            self.joints = [joint_from_name(self.robot, "x"),
                           joint_from_name(self.robot, "y"),
                           joint_from_name(self.robot, "theta")]
            set_joint_positions(self.robot, self.joints, self.start)
            self.display_goal(self.goal)

            blocking_chair = self.add_chair()

            set_joint_position(blocking_chair, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair,
                     Pose(point=Point(
                         x=2,
                         y=2.3,
                         z=0.42,
                     )
                     )
                     )
            chair_aabb = get_aabb(blocking_chair)
            self.objects_prop[blocking_chair] = [chair_aabb.upper[0] - chair_aabb.lower[0],
                                                 chair_aabb.upper[1] - chair_aabb.lower[1],
                                                 chair_aabb.upper[2] - chair_aabb.lower[2],
                                                 1]

            self.room = self.create_room(movable_obstacles=[blocking_chair])
            self.objects += [blocking_chair]
            self.push_only = []
            self.static_objects = []
            self.setup_grids()
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()

    def create_room(self, movable_obstacles=[]):
        width = 6
        length = 4
        wall_height = 2
        center = [2, 1]

        hall_width = 2
        hall_length = 3
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
                 Pose(
                     point=Point(y=center[1], x=center[0] - (width / 2 + wall_thickness / 2), z=wall_height / 2)))

        blocking_wall = create_box(1.5, 2.5, wall_height, mass=1, color=LIGHT_GREY)
        set_pose(blocking_wall,
                 Pose(point=Point(
                     x=center[0],
                     y=0.25,
                     z=wall_height/2,
                 )
                 )
                 )

        walls = [wall_1, wall_2, wall_3, wall_4, blocking_wall]
        floors = [floor1]
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, 0 + GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room