from environments.environment import Environment, Room, LIGHT_GREY, GRID_HEIGHT

from pybullet_planning.pybullet_tools.utils import (create_box, TAN, BROWN, AABB,
                                                    set_pose, Pose, Point, LockRenderer,
                                                    set_joint_position, load_model)
import math
import random
import pybullet as p


class DoubleMovable(Environment):
    def __init__(self):
        super(DoubleMovable, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (6, 2, 0)  # TODO: Create separate class for configuration space
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
            blocking_box = create_box(3, 4.5, 1, mass=1, color=BROWN)
            set_pose(blocking_box,
                     Pose(point=Point(
                         x=3,
                         y=1.25,
                         z=1 / 2,
                     )
                     )
                     )
            self.objects_prop[blocking_box] = [2, 4.5, 1, 1]

            blocking_chair_1 = load_model(
                "../models/partnet_mobility/179/mobility.urdf", scale=0.5
            )

            set_joint_position(blocking_chair_1, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair_1,
                     Pose(point=Point(
                         x=2,
                         y=4.25,
                         z=0.42,
                     )
                     )
                     )
            chair_aabb = get_aabb(blocking_chair_1)
            self.objects_prop[blocking_chair] = [chair_aabb.upper[0] - chair_aabb.lower[0],
                                                 chair_aabb.upper[1] - chair_aabb.lower[1],
                                                 chair_aabb.upper[2] - chair_aabb.lower[2],
                                                 1]

            blocking_chair_2 = load_model(
                "../models/partnet_mobility/179/mobility.urdf", scale=0.5
            )

            set_joint_position(blocking_chair_2, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair_2,
                     Pose(point=Point(
                         x=3,
                         y=4.25,
                         z=0.42,
                     )
                     )
                     )
            chair_aabb = get_aabb(blocking_chair_2)
            self.objects_prop[blocking_chair] = [chair_aabb.upper[0] - chair_aabb.lower[0],
                                                 chair_aabb.upper[1] - chair_aabb.lower[1],
                                                 chair_aabb.upper[2] - chair_aabb.lower[2],
                                                 1]

            self.room = self.create_room(movable_obstacles=[blocking_chair_1, blocking_chair_2])
            self.objects += [blocking_box, blocking_chair_1, blocking_chair_2]
            self.static_objects = [blocking_box]
            self.centered_aabb = self.get_centered_aabb()
            self.setup_grids()

    def create_room(self, movable_obstacles=[]):
        width = 10
        length = 6
        wall_height = 2
        center = [3, 2]

        hall_width = 2
        hall_length = 3
        floor1 = self.create_pillar(width=width, length=length, color=TAN)
        floor2 = self.create_pillar(width=4, length=1, color=TAN)
        floor3 = self.create_pillar(width=4, length=1, color=TAN)
        set_pose(floor1, Pose(Point(x=center[0], y=center[1])))
        set_pose(floor2, Pose(Point(x=6, y=5.5)))
        set_pose(floor3, Pose(Point(x=0, y=5.5)))

        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1,
                 Pose(point=Point(x=center[0], y=center[1] + length / 2 + wall_thickness / 2, z=wall_height / 2)))

        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2,
                 Pose(point=Point(x=center[0], y=center[1] - (length / 2 + wall_thickness / 2), z=wall_height / 2)))

        wall_3 = self.create_pillar(length=length+1, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3,
                 Pose(point=Point(y=center[1]+0.5, x=center[0] + width / 2 + wall_thickness / 2, z=wall_height / 2)))

        wall_4 = self.create_pillar(length=length+1, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4,
                 Pose(point=Point(y=center[1]+0.5, x=center[0] - (width / 2 + wall_thickness / 2), z=wall_height / 2)))

        wall_5 = self.create_pillar(width=4, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5,
                 Pose(point=Point(x=0, y=6 + (wall_thickness / 2), z=wall_height / 2)))

        wall_6 = self.create_pillar(width=4, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_6,
                 Pose(point=Point(x=6, y=6 + (wall_thickness / 2), z=wall_height / 2)))

        wall_7 = self.create_pillar(width=wall_thickness, length=1, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_7,
                 Pose(point=Point(x=2+ (wall_thickness / 2), y=5.5 , z=wall_height / 2)))

        wall_8 = self.create_pillar(width=wall_thickness, length=1, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_8,
                 Pose(point=Point(x=4 - (wall_thickness / 2), y=5.5, z=wall_height / 2)))

        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6, wall_7, wall_8]
        floors = [floor1, floor2, floor3]
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0 + 1, 0 + GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room