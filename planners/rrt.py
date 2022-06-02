from planners.planner import Planner
from utils.utils import get_pointcloud_from_camera_image
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions)
import numpy as np
from itertools import product
import time
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from matplotlib import collections  as mc


class RRT(Planner):
    def __init__(self):
        super(RRT, self).__init__()
        self.step_size = [0.05, np.pi/18]

    def get_plan(self, environment):
        environment.setup()
        
        camera_pose, image_data = environment.get_robot_vision()
        environment.update_visibility(camera_pose, image_data)
        environment.update_occupancy(image_data)

        environment.plot_grids(visibility=False, occupancy=True)


        self.joints = [joint_from_name(environment.robot, "x"),
                       joint_from_name(environment.robot, "y"),
                       joint_from_name(environment.robot, "theta")]

        start= (0,0,0)
        goal = (5,1,0)
        graph = self.rrt(start, goal, environment, n_iter=500)
        final_path = dijkstra(graph) if graph.success else None
        print(final_path)
        plot(graph, path=final_path)
        
        final_path = self.adjust_angles(final_path, start, goal)
        for q in final_path:
            set_joint_positions(environment.robot, self.joints, q)
            time.sleep(0.1)        
        wait_if_gui()




    def rrt(self, start, goal, environment, n_iter=500, radius = 0.3, goal_bias=0.1):
        lower, upper = environment.room.aabb
        G = Graph(start, goal)

        for _ in range(n_iter):
            goal_sample = np.random.choice([True, False], 1, p=[goal_bias, 1-goal_bias])[0]
            if goal_sample:
                rand_vex = goal
            else:
                rand_vex = self.sample(lower, upper)

            near_vex, near_idx = G.nearest_vex(rand_vex, environment, self.joints)

            if near_vex is None:
                continue

            new_vex = self.steer(near_vex, rand_vex)

            if environment.check_collision_in_path(self.joints, near_vex, new_vex):
                continue


            new_idx = G.add_vex(new_vex)
            dist = distance(new_vex, near_vex)
            G.add_edge(new_idx, near_idx, dist)

            dist_to_goal = distance(new_vex, G.endpos)

            if dist_to_goal < radius:
                end_idx = G.add_vex(G.endpos)
                G.add_edge(new_idx, end_idx, dist_to_goal)
                G.success = True
                return G

        return G


    def sample(self, lower_limit, upper_limit):
        rand_x = np.random.uniform(lower_limit[0], upper_limit[0])
        rand_y = np.random.uniform(lower_limit[1], upper_limit[1])
        rand_t = np.random.uniform(0, 2*np.pi)
        return (rand_x, rand_y, rand_t)


    def steer(self, source_vex, dest_vex, step_size=0.1):
        dirn = np.array(dest_vex[0:2]) - np.array(source_vex[0:2])
        length = np.linalg.norm(dirn)
        dirn = (dirn / length) * min(step_size, length)

        new_vex = (source_vex[0]+dirn[0], source_vex[1]+dirn[1], dest_vex[2])
        return new_vex


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


    def nearest_vex(self, vex, environment, joints):
        near_vex = None
        near_idx = None
        min_dist = np.inf

        for idx, v in enumerate(self.vertices):
            #if environment.check_collision_in_path(joints, v, vex):
            #    continue

            dist = distance(v, vex)

            if dist < min_dist:
                near_vex = v
                near_idx = idx
                min_dist = dist

        return near_vex, near_idx


def distance(vex1, vex2):
    return ((vex1[0] - vex2[0])**2 + (vex1[1]-vex2[1])**2)**0.5


def plot(G, path=None):
    '''
    Plot RRT, obstacles and shortest path
    '''
    px = [x for x, y, t in G.vertices]
    py = [y for x, y, t in G.vertices]
    fig, ax = plt.subplots()

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]][0:2], G.vertices[edge[1]][0:2]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i][0:2], path[i+1][0:2]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()

def dijkstra(G):
    '''
        Dijkstra algorithm for finding shortest path from start position to end.
    '''
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)
