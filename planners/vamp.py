from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, AABB, OOBB, Pose, draw_oobb, LockRenderer,
                                                    Point, draw_aabb, set_joint_positions, joint_from_name,
                                                    get_link_pose, link_from_name, get_camera_matrix, draw_pose,
                                                    multiply, tform_point, invert, pixel_from_point, get_aabb_volume,
                                                    get_aabb_vertices)
import numpy as np
import time
import datetime
import scipy.spatial
import pickle
import os

from utils.graph import Graph
from environments.vamp_environment import GRID_RESOLUTION


class Vamp(Planner):
    def __init__(self, env):
        super(Vamp, self).__init__()

        self.env = env
        self.env.setup()

        self.G = Graph()
        self.G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi/8])

        self.env.setup_default_vision(self.G)

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

        self.occupied_voxels = dict()


    def get_plan(self, loadfile=None):
        q_start, q_goal = (0, 0, 0), (6, 2, 0)
        self.v_0 = self.get_circular_vision(q_start)
        self.env.update_vision_from_voxels(self.v_0)

        self.R = self.get_circular_vision(q_goal)

        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data, q_start)
        self.env.update_occupancy(image_data)
        self.env.plot_grids(True, True, True)

        self.complete = False
        self.current_q = q_start

        if loadfile is not None:
            self.load_state("saves/" + loadfile)

            self.env.plot_grids(True, True, True)
            set_joint_positions(self.env.robot, self.joints, self.current_q)
            print("State loaded")
            wait_if_gui()

        while not self.complete:
            path = self.tourist(self.current_q, self.R, self.v_0, relaxed=False)
            #path = self.vamp_path_vis(current_q, q_goal, v_0, relaxed=False)
            if path is None:
                print("Can't find path")
                break
            print("Found path:")
            print(path)

            self.current_q, self.complete, gained_vision = self.execute_path(path)
            self.v_0.update(gained_vision)

            print("Want to save this tate? Press Y or N then Enter")
            x = input()
            if x == "Y" or x == "y":
                self.save_state()

        print("Reached the goal")
        wait_if_gui()

    def tourist(self, q_start, R, v_0, relaxed=False, obstructions=set()):
        q_goal = None
        score = 0
        for i in range(1000):
            q_rand = self.G.rand_vex()
            if not self.obstruction_from_path([q_rand], obstructions):
                new_score = len(self.env.get_optimistic_vision(q_rand, self.G).intersection(R))
                if new_score != 0:
                    if new_score > score:
                        q_goal = q_rand
                        score = new_score

        def heuristic_fn(q):
            return distance(q, q_goal)
            '''
            vision_q = self.env.get_optimistic_vision(q, self.G)
            if len(R.intersection(vision_q)) != 0:
                #return 0
                return 0 + distance(q, q_goal)
            if len(vision_q) == 0:
                return np.inf
            s1 = np.array(list(vision_q))
            s2 = np.array(list(R))

            #return scipy.spatial.distance.cdist(s1, s2).min()
            return distance(q, q_goal) + scipy.spatial.distance.cdist(s1, s2).min()*GRID_RESOLUTION
            '''
        return self.vamp_path_vis(q_start, q_goal, v_0, H=heuristic_fn, relaxed=relaxed, obstructions=obstructions)


    def vamp_step_vis(self, q_start, q_goal, v_0, H=0, relaxed=False, obstructions=set()):
        if H == 0:
            H = lambda x: distance(x, q_goal)
            # H = lambda x: 0

        return self.a_star(q_start, q_goal, v_0, H, relaxed, self.action_fn_step, obstructions=obstructions)


    def vamp_path_vis(self, q_start, q_goal, v_0, H=0, relaxed=False, obstructions=set()):
        if H == 0:
            H = lambda x: distance(x, q_goal)
            # H = lambda x: 0
        self.vision_q = dict()

        return self.a_star(q_start, q_goal, v_0, H, relaxed, self.action_fn_path, obstructions=obstructions)


    def action_fn_step(self, path, v_0, relaxed=False, extended=set(), obstructions=set()):
        actions = []
        q = path[-1]
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            if q_prime in extended:
                continue
            if relaxed:
                if not self.obstruction_from_path([q, q_prime], obstructions):
                    v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    s_q = self.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    else:
                        cost = distance(q, q_prime) *\
                                abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
                        actions.append((q_prime, cost))
            else:
                if not self.obstruction_from_path([q, q_prime], obstructions):
                    v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    s_q = self.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
        return actions


    def action_fn_path(self, path, v_0, relaxed=False, extended=set(), obstructions=set()):
        actions = []
        q = path[-1]
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            if q_prime in extended:
                continue
            if relaxed:
                if not self.obstruction_from_path([q, q_prime], obstructions):
                    if len(path) == 1:
                        v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    else:
                        v_q = self.vision_q[path[-2]].union(self.env.get_optimistic_vision(q, self.G))
                    self.vision_q[q] = v_q
                    s_q = self.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    else:
                        cost = distance(q, q_prime) *\
                                abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
                        actions.append((q_prime, cost))
            else:
                if not self.obstruction_from_path([q, q_prime], obstructions):
                    if len(path) == 1:
                        v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    else:
                        v_q = self.vision_q[path[-2]].union(self.env.get_optimistic_vision(q, self.G))
                    self.vision_q[q] = v_q
                    s_q = self.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
        return actions


    def get_circular_vision(self, q, radius=1):
        grid = self.env.static_vis_grid
        surface_aabb = grid.aabb
        voxels = set()
        for voxel in grid.voxels_from_aabb(surface_aabb):
            actual_q = (q[0], q[1], 0)
            actual_vox = np.array(voxel) * np.array(self.G.res)
            if distance(actual_vox, actual_q) < radius:
                voxels.add(voxel)
        return voxels


    def visibility_voxels_from_path(self, path):
        voxels = set()
        for q in path:
            voxel = [x for x in self.env.static_vis_grid.voxels_from_aabb(self.env.aabb_from_q(q))
                        if self.env.static_vis_grid.contains(x)]
            voxels.update(voxel)
        return voxels


    def obstruction_from_path(self, path, obstruction):
        voxels = set()

        for q in path:

            voxel = [x for x in self.env.occupancy_grid.voxels_from_aabb(self.env.aabb_from_q(q))
                     if self.env.occupancy_grid.contains(x)]
            voxels.update(voxel)

            voxel = obstruction.intersection(self.visibility_voxels_from_path([q]))
            voxels.update(voxel)

        return voxels


    def volume_from_voxels(self, grid, voxels):
        if len(voxels) == 0:
            return 0
        voxel_vol = get_aabb_volume(grid.aabb_from_voxel(next(iter(voxels))))
        return voxel_vol*len(voxels)



    def a_star(self, q_start, q_goal, v_0, H, relaxed, action_fn, obstructions=set()):
        current_t = time.clock_gettime_ns(0)
        extended = set()
        paths = [([q_start], 0, 0)]

        while paths:
            current = paths.pop(-1)
            best_path = current[0]
            best_path_cost = current[1]


            if best_path[-1] in extended:
                continue

            if best_path[-1] == q_goal:
                done = time.clock_gettime_ns(0) - current_t
                print(done * (10 ** (-9)))
                self.G.plot_search(self.env, extended, path=best_path)
                return best_path

            extended.add(best_path[-1])
            for action in action_fn(best_path, v_0, relaxed=relaxed, extended=extended, obstructions=obstructions):
                paths.append((best_path + [action[0]], best_path_cost+ action[1], H(action[0])))

            # Only sorting from heuristic. Faster but change if needed
            paths = sorted(paths, key=lambda x: x[-1], reverse=True)


        return None


    def execute_path(self, path):
        gained_vision = set()
        for qi, q in enumerate(path):
            next_occupied = self.visibility_voxels_from_path([q])
            for voxel in next_occupied:
                if self.env.visibility_grid.contains(voxel):
                    qi = qi-1 if qi-1 >= 0 else 0
                    print("Stepping into unseen area. Aborting")
                    return path[qi], False, gained_vision
            set_joint_positions(self.env.robot, self.joints, q)

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(image_data)
            self.env.update_movable_boxes(image_data)
            gained_vision.update(self.env.update_visibility(camera_pose, image_data, q))

            # Check if remaining path is collision free under the new occupancy grid
            if len(self.obstruction_from_path(path[qi:], set())) != 0:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                return q, False, gained_vision
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)
        return q, True, gained_vision


    def save_state(self):
        current_time = datetime.datetime.now()
        dbfile = open("saves/state_{}_{}_{}_{}_{}.dat".format(current_time.month, current_time.day, current_time.hour,
                                           current_time.minute, current_time.second), "wb")
        pickle.dump(self, dbfile)
        dbfile.close()

    def load_state(self, filename):
        dbfile = open(filename, 'rb')
        copy = pickle.load(dbfile)

        self.env = copy.env
        self.G = copy.G
        self.occupied_voxels = copy.occupied_voxels
        self.v_0 = copy.v_0

        self.R = copy.R


        self.complete = copy.complete
        self.current_q = copy.current_q

        dbfile.close()


def distance(vex1, vex2):
    dist = 0
    for i in range(len(vex1)-1):
        dist += (vex1[i] - vex2[i])**2
    #dist += (vex1[len(vex1)-1] - vex2[len(vex1)-1])**2/100
    return dist**0.5

