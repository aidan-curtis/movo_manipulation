from environments.environment import Environment, Room, GRID_HEIGHT, LIGHT_GREY, GRID_RESOLUTION
from pybullet_planning.pybullet_tools.utils import (set_pose, set_joint_position, Pose, Point,
                                                    load_model, TAN, RED,
                                                    LockRenderer, AABB, get_aabb, joint_from_name,
                                                    set_joint_positions)
import random
import math
import pybullet as p
import numpy as np
   

class SubObs(Environment):
    def __init__(self, **kwargs):
        super(SubObs, self).__init__(**kwargs)

        self.start = (0, 0, 0)
        self.goal = (5.4, -0.4, 0)
        self.chair_pos = (4, 3.5, 0.42)


        self.objects = []
        self.viewed_voxels = []

        # Properties represented as a list of width, length, height, mass
        self.objects_prop = dict()
        self.initialized = False

    def setup(self):

        self.disconnect()
        self.connect()

        with LockRenderer():

            # These 3 lines are important and should be located here
            self.robot = self.setup_robot()
            self.centered_aabb = self.get_centered_aabb()
            self.centered_oobb = self.get_centered_oobb()

            if not self.initialized:
                self.randomize_env()
            self.display_goal(self.goal)

            self.joints = [joint_from_name(self.robot, "x"),
                        joint_from_name(self.robot, "y"),
                        joint_from_name(self.robot, "theta")]
            set_joint_positions(self.robot, self.joints, self.start)
            

            blocking_chair = self.add_chair()
            
            self.room = self.create_room(movable_obstacles=[blocking_chair])
            set_joint_position(blocking_chair, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair,
                Pose(point=Point(
                    x=self.chair_pos[0],
                    y=self.chair_pos[1],
                    z=self.chair_pos[2],
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
        width = 7
        length = 6
        wall_height = 2
        center = [2.5, 2]

        floor1 = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor1, Pose(Point(x=center[0], y=center[1])))

        floor2 = self.create_pillar(width=1.9, length=3.8, color=TAN)
        set_pose(floor2, Pose(Point(x=4 + 0.05, y=0.9, z=0.001)))

        wall_thickness = 0.1
        # Left wall
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1,
                 Pose(point=Point(x=center[0], y=4.95, z=wall_height / 2)))

        # Right wall
        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2,
                 Pose(point=Point(x=center[0], y=-0.95, z=wall_height / 2)))

        # Front wall
        wall_3 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3,
                 Pose(point=Point(y=center[1], x=5.95, z=wall_height / 2)))

        # Back wall
        wall_4 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4,
                 Pose(point=Point(y=center[1], x=-0.95, z=wall_height / 2)))

        # Dividing wall
        wall_5 = self.create_pillar(length=3.9, width=wall_thickness*2, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5,
                 Pose(point=Point(y=0.95, x=3, z=wall_height / 2)))

        # Miniature wall
        wall_6 = self.create_pillar(length=5.1, width=wall_thickness, height=0.7, color=LIGHT_GREY)
        set_pose(wall_6,
                 Pose(point=Point(y=1.55, x=4.749, z=0.351)))

        wall_7 = self.create_pillar(length=5.1, width=wall_thickness, height=0.39, color=LIGHT_GREY)
        set_pose(wall_7,
                 Pose(point=Point(y=1.55, x=4.85, z=0.1951)))


        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6, wall_7]
        floors = [floor1, floor2]
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room

    def randomize_env(self):
        i = np.random.randint(6)
        e = np.random.randint(6)
        self.start = (round(self.start[0] + i*GRID_RESOLUTION, 2),
                      round(self.start[1] + e*GRID_RESOLUTION, 2),
                      round(self.start[2] + np.random.randint(16)*np.pi/8, 3))

        i = np.random.randint(0, 3)
        self.goal = (self.goal[0],
                     round(self.goal[1] + i*GRID_RESOLUTION, 2),
                     self.goal[2])

        i = np.random.randint(-5, 0)
        self.chair_pos = (self.chair_pos[0],
                          self.chair_pos[1] + i*0.2,
                          self.chair_pos[2])

        self.initialized = True
