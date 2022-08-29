from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions,
                                                    draw_aabb, AABB, get_aabb)
import numpy as np
from itertools import product
import time
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.patches import Rectangle, Arrow


class AStarSearch(Planner):
    def __init__(self, env):
        super(AStarSearch, self).__init__()
        self.env = env
        self.step_size = [0.1, 0.1]

        self.env.setup()

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]


    def get_plan(self, **kwargs):

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

class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos: 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.0}


    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx


    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


    def nearest_vex(self, vex, environment, joints, k=0):
        neighbors = [(distance(v,vex), v, idx) for idx,v in enumerate(self.vertices)]
        neighbors = sorted(neighbors, key=lambda x: x[0])

        if len(neighbors) < k+1:
            return None, None

        return neighbors[k][1], neighbors[k][2]

    def initialize_full_graph(self, env, resolution=[0.1, 0.1, 0]):
        x_step = int((env.room.aabb.upper[0] - env.room.aabb.lower[0])/resolution[0])+1
        y_step = int((env.room.aabb.upper[1] - env.room.aabb.lower[1])/resolution[1])+1
        t_step = int((2*np.pi)/(resolution[2])) if resolution[2] != 0 else 1

        x = env.room.aabb.lower[0]-resolution[0]
        for i in range(x_step):
            y = env.room.aabb.lower[1]-resolution[1]
            x += resolution[0]
            for j in range(y_step):
                y+= resolution[1]
                t = -resolution[2]
                for k in range(t_step):
                    t+= resolution[2]
                    self.add_vex((x,y,t))


    def dijkstra(self):
        '''
            Dijkstra algorithm for finding shortest path from start position to end.
        '''
        srcIdx = self.vex2idx[self.startpos]
        dstIdx = self.vex2idx[self.endpos]

        # build dijkstra
        nodes = list(self.neighbors.keys())
        dist = {node: float('inf') for node in nodes}
        prev = {node: None for node in nodes}
        dist[srcIdx] = 0

        while nodes:
            curNode = min(nodes, key=lambda node: dist[node])
            nodes.remove(curNode)
            if dist[curNode] == float('inf'):
                break

            for neighbor, cost in self.neighbors[curNode]:
                newCost = dist[curNode] + cost
                if newCost < dist[neighbor]:
                    dist[neighbor] = newCost
                    prev[neighbor] = curNode

        # retrieve path
        path = deque()
        curNode = dstIdx
        while prev[curNode] is not None:
            path.appendleft(self.vertices[curNode])
            curNode = prev[curNode]
        path.appendleft(self.vertices[curNode])
        return list(path)


    def plot(self, env, path=None):
        '''
        Plot RRT, obstacles and shortest path
        '''
        px = [x for x, y, t in self.vertices]
        py = [y for x, y, t in self.vertices]
        pt = [t for x, y, t in self.vertices]
        fig, ax = plt.subplots()

        ax.scatter(px, py, c='cyan')
        ax.scatter(self.startpos[0], self.startpos[1], c='black')
        ax.scatter(self.endpos[0], self.endpos[1], c='red')

        lines = [(self.vertices[edge[0]][0:2], self.vertices[edge[1]][0:2]) for edge in self.edges]
        for points in lines:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            ax.add_patch(Arrow(points[0][0], points[0][1], dx, dy, width=0.01))
        #lc = mc.LineCollection(lines, colors='green', linewidths=2)
        #ax.add_collection(lc)

        # Draw angles of points
        angle_lines = []
        for x,y,t in self.vertices:
            endy = y + 0.05 * np.sin(t)
            endx = x+ 0.05 * np.cos(t)
            angle_lines.append(((x,y), (endx, endy)))
        lc = mc.LineCollection(angle_lines, colors='red', linewidths=2)
        ax.add_collection(lc)

        # Draw room shape
        for wall in env.room.walls:
            wall_aabb = get_aabb(wall)
            rec = Rectangle((wall_aabb.lower[0:2]),
                         wall_aabb.upper[0] - wall_aabb.lower[0],
                         wall_aabb.upper[1] - wall_aabb.lower[1],
                         color="grey", linewidth=0.1)
            ax.add_patch(rec)

        # Not taking rotations into account
        for obstacle in env.static_objects + env.movable_boxes:
            color = "brown"
            if isinstance(obstacle, int):
                aabb = get_aabb(obstacle)
            else:
                aabb = obstacle.aabb
                color = "yellow"
            ax.add_patch(Rectangle((aabb.lower[0], aabb.lower[1]),
                         aabb.upper[0] - aabb.lower[0],
                         aabb.upper[1] - aabb.lower[1],
                         color=color, linewidth=0.1))


        if path is not None:
            paths = [(path[i][0:2], path[i+1][0:2]) for i in range(len(path)-1)]
            lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
            ax.add_collection(lc2)

        ax.autoscale()
        ax.margins(0.1)
        plt.show()

def distance(vex1, vex2):
    return ((vex1[0] - vex2[0])**2 + (vex1[1]-vex2[1])**2)**0.5





