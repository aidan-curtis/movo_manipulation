
from planners.rrt import RRT 
from pybullet_planning.pybullet_tools.utils import (LockRenderer, wait_if_gui, joint_from_name, set_joint_positions)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections  as mc


class Snowplow(RRT):
    def __init__(self, env):
        super(Snowplow, self).__init__(env)
        self.env = env
        
        # Setup the environment
        self.env.setup()

        self.step_size = [0.05, np.pi/18]
        self.RRT_ITERS = 5000

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]


    def get_plan(self):
        
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data)
        self.env.update_occupancy(image_data)

        self.env.plot_grids(visibility=False, occupancy=True, movable=True)


        current_q, complete = self.env.start, False

        while(not complete):
            final_path = self.get_path(current_q, self.env.goal)
            if(final_path is None):
                print("No direct path to goal")
                relaxed_final_path = self.get_path(current_q, self.env.goal, ignore_movable=True)
                if(relaxed_final_path is None):
                    print("No indirect path to goal :(")
                else:
                    print(relaxed_final_path)
                import sys
                sys.exit()

                
            current_q, complete = self.execute_path(final_path)

        wait_if_gui()
    
