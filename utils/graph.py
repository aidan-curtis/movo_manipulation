import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.patches import Rectangle, Arrow
import numpy as np

from pybullet_planning.pybullet_tools.utils import get_aabb

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
