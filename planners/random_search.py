from planners.planner import Planner
from utils.utils import get_pointcloud_from_camera_image
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions)
import numpy as np
from itertools import product
import time


class RandomSearch(Planner):
    def __init__(self):
        super(RandomSearch, self).__init__()
        self.step_size = [0.05, np.pi/18]

    def get_plan(self, environment):
        environment.setup()
        
        camera_pose, image_data = environment.get_robot_vision()
        environment.update_visibility(camera_pose, image_data)
        environment.update_occupancy(image_data)

        environment.plot_grids(visibility=False, occupancy=False)


        self.joints = [joint_from_name(environment.robot, "x"),
              joint_from_name(environment.robot, "y"),
              joint_from_name(environment.robot, "theta")]

        q = [0.96, 0, 0]

        final_path = self.search_Astar((0,0,0), (1,1,0), environment)
        print(final_path)

        for q in final_path:
            set_joint_positions(environment.robot, self.joints, q)
            time.sleep(0.1)

        wait_if_gui()


    def distance_between_nodes(self, node1, node2):
        return ((node1[0] - node2[0])**2 + (node1[1]-node2[1])**2)**0.5
    

    def angle_distance_between_nodes(self, node1, node2):
        return (node1[2] - node2[2])**2

    def compute_heuristics(self, current, goal):
        h = self.distance_between_nodes(current, goal)
        #h += abs(current[2] - goal[2])**0.1
        return h


    def extend(self, node):
        # Only able to either move forward or rotate to keep visibility constraint
        x_step = np.cos(node[2])*self.step_size[0]
        y_step = np.sin(node[2])*self.step_size[0]

        new_pos = [(node[0] + x_step, node[1] + y_step, node[2]),
                   (node[0] - x_step, node[1] - y_step, node[2]),
                   (node[0], node[1], round((node[2] + self.step_size[1])%(2*np.pi),5)),
                   (node[0], node[1], round((node[2] - self.step_size[1])%(2*np.pi),5))]


        return new_pos


    def check_end(self, current, goal, threshold= [0.05, 0.05, 0.01]):
        for i in range(len(current)):
            if  i == 2:
                if abs(current[i] - goal[i]) > threshold[i] and abs(current[i] - 2*np.pi) > threshold[i]:
                    return False
            elif abs(current[i] - goal[i]) > threshold[i]:
                return False
        return True



    def search_Astar(self, start, goal, environment):
        paths = [[[start], 0, 0]]
        extended = set()
        while paths:

            path = paths.pop(0)
            current = path[0]
            if current[-1] in extended:
                continue
            if self.check_end(current[-1], goal):
                return current

            
            extended.add(current[-1])
            new_nodes = self.extend(current[-1])
            new_paths = []
            for node in new_nodes:
                if not environment.check_state_collision(self.joints, node):
                    paths.append([current + [node], self.compute_heuristics(node, goal), 
                        path[-1] + self.distance_between_nodes(current[-1], node) + self.angle_distance_between_nodes(current[-1], node)])
            paths = sorted(paths, key=lambda x: x[-2])
        return None




