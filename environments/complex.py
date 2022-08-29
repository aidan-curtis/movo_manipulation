#from environments.environment import Environment, Room, GRID_HEIGHT, LIGHT_GREY
from environments.vamp_environment import Environment, Room, GRID_HEIGHT, LIGHT_GREY


from pybullet_planning.pybullet_tools.utils import (set_pose, set_joint_position, Pose, Point,
                                                    load_model, create_box, TAN, BROWN,
                                                    LockRenderer, AABB, get_aabb)
import random
import math
import pybullet as p
import numpy as np
   

class Complex(Environment):
    def __init__(self):
        super(Complex, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (2, -4, round(np.pi/2, 3)) # TODO: Create separate class for configuration space

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

            blocking_chair = load_model(
                    "../models/partnet_mobility/179/mobility.urdf", scale=0.5
                )
            blocking_box = create_box(1, 2.1, 1, mass=1, color=BROWN)
            self.room = self.create_room(movable_obstacles=[blocking_chair, blocking_box])
            set_joint_position(blocking_chair, 17, random.uniform(-math.pi, math.pi))
            set_pose(blocking_chair,
                Pose(point=Point(
                        x=2,
                        y=1,
                        z=0.42,
                    )
                )
            )

            set_pose(blocking_box,
                     Pose(point=Point(
                         x=2,
                         y=-0.72,
                         z=1 / 2,
                     )
                     )
                     )
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
        width = 6
        length = 4
        wall_height = 2
        center = [2,0]

        hall_width = 2
        hall_length = 3
        floor1 = self.create_pillar(width=width, length=length-0.2, color=TAN)
        floor2 = self.create_pillar(width=hall_width, length=hall_length, color=TAN)
        set_pose(floor1, Pose(Point(x=center[0], y=center[1]-0.1)))
        set_pose(floor2, Pose(Point(x=2, y=-3.5)))


        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1, Pose(point=Point(x=center[0], y=center[1]+length/2+wall_thickness/2-0.21, z=wall_height/2)))
        
        wall_2 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2, Pose(point=Point(x=0, y=center[1]-(length/2+wall_thickness/2), z=wall_height/2)))
        
        wall_3 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3, Pose(point=Point(x=4, y=center[1]-(length/2+wall_thickness/2), z=wall_height/2)))

        wall_4 = self.create_pillar(length=length-0.21, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4, Pose(point=Point(y=center[1]-0.21/2, x=center[0]+width/2+wall_thickness/2, z=wall_height/2)))
        
        wall_5 = self.create_pillar(length=length-0.21, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_5, Pose(point=Point(y=center[1]-0.21/2, x=center[0]-(width/2+wall_thickness/2), z=wall_height/2)))

        wall_6 = self.create_pillar(width=2, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_6, Pose(point=Point(x=2, y=center[1]-(length/2+wall_thickness/2+3), z=wall_height/2)))

        wall_7 = self.create_pillar(length=3, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_7, Pose(point=Point(y=-3.5, x=1-wall_thickness/2, z=wall_height/2)))

        wall_8 = self.create_pillar(length=3, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_8, Pose(point=Point(y=-3.5, x=3+wall_thickness/2, z=wall_height/2)))

        walls = [wall_1, wall_2, wall_3, wall_4, wall_5, wall_6, wall_7, wall_8]
        floors = [floor1, floor2]
        aabb = AABB(lower=(center[0]-width/2.0, center[1]-length/2.0-hall_length, 0.05), 
                    upper=(center[0]+width/2.0, center[1]+length/2.0, 0 + GRID_HEIGHT))
        room = Room(walls, floors, aabb, movable_obstacles)

        return room
