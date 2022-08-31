from environments.environment import Environment, Room, GRID_HEIGHT, LIGHT_GREY
from pybullet_planning.pybullet_tools.utils import (set_pose, set_joint_position, Pose, Point,
                                                    load_model, TAN, RED,
                                                    LockRenderer, AABB, get_aabb, joint_from_name,
                                                    set_joint_positions)
import random
import math
import pybullet as p
import numpy as np
   

class SubObs(Environment):
    def __init__(self):
        super(SubObs, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (4, 0, round(np.pi/2, 3)) # TODO: Create separate class for configuration space


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

        # These 3 lines are important and should be located here
        self.robot = self.setup_robot()
        self.centered_aabb = self.get_centered_aabb()
        self.centered_oobb = self.get_centered_oobb()

        self.joints = [joint_from_name(self.robot, "x"),
                       joint_from_name(self.robot, "y"),
                       joint_from_name(self.robot, "theta")]
        set_joint_positions(self.robot, self.joints, self.start)

        with LockRenderer():

            blocking_chair = load_model(
                    "../models/partnet_mobility/179/mobility.urdf", scale=0.5
                )
            self.room = self.create_room(movable_obstacles=[blocking_chair])
            set_joint_position(blocking_chair, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair,
                Pose(point=Point(
                        x=2,
                        y=3,
                        z=0.42,
                    )
                )
            )
            chair_aabb = get_aabb(blocking_chair)
            self.objects_prop[blocking_chair] = [chair_aabb.upper[0] - chair_aabb.lower[0],
                                                 chair_aabb.upper[1] - chair_aabb.lower[1],
                                                 chair_aabb.upper[2] - chair_aabb.lower[2],
                                                 1]


            self.objects +=  [blocking_chair]
            self.push_only = []
            self.static_objects = []
            self.setup_grids()


    def create_room(self, movable_obstacles=[]):
        width = 6
        length = 6
        wall_height = 2
        center = [2, 2]

        floor1 = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor1, Pose(Point(x=center[0], y=center[1])))

        floor2 = self.create_pillar(width=1.9, length=3.8, color=RED)
        set_pose(floor2, Pose(Point(x=4+0.05, y=0.9, z=0.001)))

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
                 Pose(point=Point(y=0.95, x=1.1, z=wall_height / 2)))

        wall_6 = self.create_pillar(length=3.9, width=wall_thickness, height=0.1, color=LIGHT_GREY)
        set_pose(wall_6,
                 Pose(point=Point(y=0.95, x=3.1, z=0.05)))

        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6]
        floors = [floor1, floor2]
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room

    def restrict_configuration(self, G):
        aabb = AABB(lower=[3.1, -1.1, 0], upper=[5.1, 2.9, 8])
        idxs = np.all((aabb.lower <= np.array(G.vertices)) & (np.array(G.vertices) <= aabb.upper), axis=1)
        for vex in np.array(G.vertices)[idxs]:
            if vex[2] != round(np.pi / 2, 3):
                G.dettach_vex(tuple(vex))
