from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions)
import numpy as np
from itertools import product
import time

class AStarSearch(Planner):
    def __init__(self):
        super(AStarSearch, self).__init__()
        self.step_size = [0.1, 0.1]

    def get_plan(self, environment):
        environment.setup()
        
        camera_pose, image_data = environment.get_robot_vision()
        environment.update_visibility(camera_pose, image_data)
        environment.update_occupancy(image_data)

        environment.plot_grids(visibility=False, occupancy=True)


        self.joints = [joint_from_name(environment.robot, "x"),
              joint_from_name(environment.robot, "y"),
              joint_from_name(environment.robot, "theta")]

        start = tuple(environment.start)
        goal = tuple(environment.goal)

        final_path = self.search_Astar(start, goal, environment)
        final_path = self.adjust_angles(final_path, start, goal)
        print(final_path)

        for q in final_path:
            set_joint_positions(environment.robot, self.joints, q)
            time.sleep(0.1)

        wait_if_gui()


    def distance_between_nodes(self, node1, node2):
        return ((node1[0] - node2[0])**2 + (node1[1]-node2[1])**2)**0.5
    

    def compute_heuristics(self, current, goal):
        h = self.distance_between_nodes(current, goal)
        return h


    def extend(self, node):
        # Only able to either move forward or rotate to keep visibility constraint
        new_pos_x = [round(node[0] + self.step_size[0],2), node[0], round(node[0] - self.step_size[0],2)]
        new_pos_y = [round(node[1] + self.step_size[1],2), node[1], round(node[1] - self.step_size[1],2)]
        new_pos_t = [0]

        return list(product(*[new_pos_x, new_pos_y, new_pos_t]))


    def check_end(self, current, goal, threshold= [0.05, 0.05]):
        for i in range(len(current)-1):
            if abs(current[i] - goal[i]) > threshold[i]:
                return False
        return True



    def search_Astar(self, start, goal, environment):
        paths = [[[start], 0.0, 0.0]]
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
                if not environment.check_state_collision(self.joints, node) and node not in extended:
                    paths.append([current + [node], self.compute_heuristics(node, goal), 
                        path[-1] + self.distance_between_nodes(current[-1], node)])
            paths = sorted(paths, key=lambda x: x[-2] + x[-1])
        return None



    def adjust_angles(self, path, start, goal):
        final_path = [start]
        for i in range(1, len(path)):
            beg = path[i-1]
            end = path[i]

            delta_x = end[0] - beg[0]
            delta_y = end[1] - beg[1]
            theta = np.arctan2(delta_y, delta_x)

            final_path.append((beg[0], beg[1], theta))
            final_path.append((end[0], end[1], theta))
        final_path.append(goal)
        return final_path




