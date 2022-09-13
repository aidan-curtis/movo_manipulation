from environments.environment import Environment, Room, GRID_HEIGHT, LIGHT_GREY, GRID_RESOLUTION

from pybullet_planning.pybullet_tools.utils import (RGBA, set_pose, set_joint_position, Pose, Point,
                                                    load_model, create_box, TAN, BROWN,
                                                    LockRenderer, AABB, get_aabb, joint_from_name, set_joint_positions)
import random
import math
import pybullet as p
import numpy as np
   

class Complex(Environment):
    def __init__(self, **kwargs):
        super(Complex, self).__init__(**kwargs)

        self.start = (0, 0, 0)
        self.goal = (2, -4, round(np.pi/2, 3))
        self.chair_pos = (2, 1, 0.42)

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
            blocking_chair = self.add_chair()

            self.joints = [joint_from_name(self.robot, "x"),
                           joint_from_name(self.robot, "y"),
                           joint_from_name(self.robot, "theta")]
            set_joint_positions(self.robot, self.joints, self.start)
                

            LIGHT_BROWN = RGBA(0.596, 0.463, 0.329, 1)
            blocking_box = create_box(1, 2.1, 1, mass=1, color=LIGHT_BROWN)
            self.room = self.create_room(movable_obstacles=[blocking_chair, blocking_box])
            
            set_pose(blocking_chair,
                Pose(point=Point(
                        x=self.chair_pos[0],
                        y=self.chair_pos[1],
                        z=self.chair_pos[2],
                    )
                )
            )

            set_pose(blocking_box, Pose(point=Point(x=2, y=-0.72, z=1 / 2)))

            chair_aabb = get_aabb(blocking_chair)
            self.objects_prop[blocking_chair] = [chair_aabb.upper[0] - chair_aabb.lower[0],
                                                 chair_aabb.upper[1] - chair_aabb.lower[1],
                                                 chair_aabb.upper[2] - chair_aabb.lower[2],
                                                 1]


            self.objects_prop[blocking_box] = [2, 4.5, 1, 1]
            self.objects += [blocking_box, blocking_chair]
            self.push_only = [blocking_box]
            self.static_objects = []
            self.setup_grids()
            

    def create_room(self, movable_obstacles=[]):
        width = 6
        length = 4
        wall_height = 2
        center = [2, 0]

        hall_width = 2
        hall_length = 3
        floor1 = self.create_pillar(width=width, length=length-0.2, color=TAN)
        floor2 = self.create_pillar(width=hall_width, length=hall_length, color=TAN)
        set_pose(floor1, Pose(Point(x=center[0], y=center[1]-0.1)))
        set_pose(floor2, Pose(Point(x=2, y=-3.5)))


        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1, Pose(point=Point(x=center[0], y=center[1]+length/2-0.25, z=wall_height/2)))

        wall_2 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2, Pose(point=Point(x=0, y=center[1]-(length/2+wall_thickness/2), z=wall_height/2)))

        wall_3 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3, Pose(point=Point(x=4, y=center[1]-(length/2+wall_thickness/2), z=wall_height/2)))

        wall_4 = self.create_pillar(length=length-0.21, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4, Pose(point=Point(y=center[1]-0.21/2, x=center[0]+width/2-wall_thickness/2, z=wall_height/2)))
        
        wall_5 = self.create_pillar(length=length-0.21, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5, Pose(point=Point(y=center[1]-0.21/2, x=center[0]-(width/2-wall_thickness/2), z=wall_height/2)))

        wall_6 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_6, Pose(point=Point(x=2, y=center[1]-(length/2+wall_thickness/2+3), z=wall_height/2)))

        wall_7 = self.create_pillar(length=3, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_7, Pose(point=Point(y=-3.5, x=1-wall_thickness/2, z=wall_height/2)))

        wall_8 = self.create_pillar(length=3, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_8, Pose(point=Point(y=-3.5, x=3+wall_thickness/2, z=wall_height/2)))

        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6, wall_7, wall_8]
        floors = [floor1, floor2]
        aabb = AABB(lower=(center[0]-width/2.0, center[1]-length/2.0-hall_length, 0.05), 
                    upper=(center[0]+width/2.0, center[1]+length/2.0-0.2, 0 + GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room



    def randomize_env(self):
        i = np.random.randint(4)
        e = np.random.randint(-4, 3)
        self.start = (round(self.start[0] + i*GRID_RESOLUTION, 2),
                      round(self.start[1] + e*GRID_RESOLUTION, 2),
                      round(self.start[2] + np.random.randint(16)*np.pi/8, 3))

        i = np.random.randint(0, 5)
        self.goal = (self.goal[0],
                     round(self.goal[1] + i*GRID_RESOLUTION, 2),
                     self.goal[2])

        i = np.random.randint(-4, 5)
        self.chair_pos = (self.chair_pos[0] + i*0.1,
                          self.chair_pos[1],
                          self.chair_pos[2])

        self.initialized = True


