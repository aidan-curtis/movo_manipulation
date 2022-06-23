from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions,
                                                    draw_aabb, AABB)
from utils.graph import Graph
import numpy as np
from itertools import product
import time

class AStarSearch(Planner):
    def __init__(self, env):
        super(AStarSearch, self).__init__()
        self.env = env
        self.step_size = [0.1, 0.1]

        self.env.setup()

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]


    def get_plan(self):

        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data)
        self.env.update_occupancy(image_data)

        current_q, complete = self.env.start, False
        self.env.plot_grids(visibility=True, occupancy=True, movable=True)

        while not complete:
            final_path = self.get_path(current_q, self.env.goal)

            current_q, complete = self.execute_path(final_path)

        wait_if_gui()


    def get_path(self, start, goal, vis=False, ignore_movable=False, forced_obj_coll=[],
                 attached_object=None, moving_backwards=False):

        path, G = self.search_Astar(start, goal,
                                 ignore_movable=ignore_movable,
                                 forced_object_coll=forced_obj_coll,
                                 attached_object=attached_object,
                                 moving_backwards=moving_backwards)
        G.plot(self.env, path=path)

        if not G.success:
            return None

        if not moving_backwards:
            final_path = self.env.adjust_angles(path, start, goal)
        else:
            final_path = self.adjust_angles(path, goal, start)
            final_path.reverse()

        return final_path



    def execute_path(self, path, ignore_movable=False):
        for qi, q in enumerate(path):
            set_joint_positions(self.env.robot, self.joints, q)

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(image_data)
            self.env.update_movable_boxes(image_data)
            self.env.update_visibility(camera_pose, image_data)

            # Check if remaining path is collision free under the new occupancy grid
            for next_qi in path[qi:]:
                if (self.env.check_conf_collision(next_qi, ignore_movable=ignore_movable)):
                    self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                    return q, False
        return q, True


    def distance_between_nodes(self, node1, node2):
        return ((node1[0] - node2[0])**2 + (node1[1]-node2[1])**2)**0.5
    

    def compute_heuristics(self, current, goal):
        h = self.distance_between_nodes(current, goal)
        return h


    def extend(self, node):
        # Only able to either move forward or rotate to keep visibility constraint
        new_pos_x = [round(node[0] + self.step_size[0] * i, 2) for i in [-1, 0, 1]]
        new_pos_x = [x for x in new_pos_x
                     if x >= self.env.room.aabb.lower[0] and x <= self.env.room.aabb.upper[0]]
        new_pos_y = [round(node[1] + self.step_size[1] * i, 2) for i in [-1, 0, 1]]
        new_pos_y = [y for y in new_pos_y
                     if y >= self.env.room.aabb.lower[1] and y <= self.env.room.aabb.upper[1]]
        new_pos_t = [0]

        return list(product(*[new_pos_x, new_pos_y, new_pos_t]))


    def check_end(self, current, goal, threshold= [0.08, 0.08]):
        for i in range(len(current)-1):
            if abs(current[i] - goal[i]) > threshold[i]:
                return False
        return True



    def search_Astar(self, start_node, goal_node,
                     ignore_movable=False, forced_object_coll=[],
                     attached_object=None, moving_backwards=False):
        start, goal = start_node, goal_node
        if moving_backwards:
            goal, start = start, goal

        G = Graph(start, goal)

        paths = [[[start], 0.0, 0.0]]
        extended = set()
        while paths:
            path = paths.pop(0)
            current = path[0]
            if current[-1] in extended:
                continue

            if self.check_end(current[-1], goal):
                G.success = True
                current+= [goal_node]
                return current, G

            extended.add(current[-1])
            new_nodes = self.extend(current[-1])
            new_paths = []
            for node in new_nodes:
                if node not in extended:
                    bad = self.env.check_collision_in_path(current[-1], node, ignore_movable=ignore_movable,
                                                        forced_object_coll=forced_object_coll,
                                                        attached_object=attached_object,
                                                        moving_backwards=False)
                    if not bad:
                        dist = self.distance_between_nodes(current[-1], node)
                        paths.append([current + [node], self.compute_heuristics(node, goal),
                            path[-1] + dist])
                        new_idx = G.add_vex(node)
                        G.add_edge(G.vex2idx[current[-1]], new_idx, dist)
                    else:
                        extended.add(node)
            paths = sorted(paths, key=lambda x: x[-2] + x[-1])
        return None,G



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




