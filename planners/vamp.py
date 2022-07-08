from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, AABB, OOBB, Pose, draw_oobb, LockRenderer,
                                                    Point, draw_aabb, set_joint_positions, joint_from_name,
                                                    get_link_pose, link_from_name, get_camera_matrix, draw_pose,
                                                    multiply, tform_point, invert, pixel_from_point, get_aabb_volume)
import numpy as np

from utils.graph import Graph


class Vamp(Planner):
    def __init__(self, env):
        super(Vamp, self).__init__()

        self.env = env
        self.env.setup()

        self.G = Graph()
        self.G.initialize_full_graph(self.env)

        self.env.setup_default_vision(self.G)

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]


    def get_plan(self):
        q_start, q_goal = (0, 0, 0), (1 ,1, round(np.pi, 3))
        v_0 = self.get_circular_vision(q_start)

        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data, q_start)
        self.env.update_occupancy(image_data)
        self.env.plot_grids(True, True, True)

        complete = False
        current_q = q_start
        while not complete:
            path = self.vamp_step_vis(current_q, q_goal, v_0, relaxed=False)
            if path is None:
                print("Can't find path")
                break
            print("Found path:")
            print(path)

            current_q, complete = self.execute_path(path)

        print("Reached the goal")
        wait_if_gui()


    def vamp_step_vis(self, q_start, q_goal, v_0, H=0, relaxed=False):
        if H ==0:
            H = lambda x: distance(x, q_goal)

        return self.a_star(q_start, q_goal, v_0, H, relaxed)


    def action_cost_fn(self, q, v_0, relaxed=False):
        actions = []
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            if relaxed:
                if not self.obstruction_from_path([q, q_prime]):
                    v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
                    s_q = self.visibility_voxels_from_path([q, q_prime])
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    else:
                        cost = distance(q, q_prime) *\
                                abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
                        actions.append((q_prime, cost))
            else:
                v_q = v_0.union(self.env.get_optimistic_vision(q, self.G))
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
                self.env.visibility_grid.set_free(voxel)
                voxels.add(voxel)
        return voxels



    def visibility_voxels_from_path(self, path):
        voxels = set()
        for q in path:
            voxel = [x for x in self.env.static_vis_grid.voxels_from_aabb(self.env.aabb_from_q(q))
                    if self.env.static_vis_grid.contains(x)]
            voxels.update(voxel)
        return voxels


    def obstruction_from_path(self, path):
        voxels = set()
        for q in path:
            voxel = [x for x in self.env.occupancy_grid.voxels_from_aabb(self.env.aabb_from_q(q))
                     if self.env.occupancy_grid.contains(x)]
            voxels.update(voxel)
        return voxels


    def volume_from_voxels(self, grid, voxels):
        if len(voxels) == 0:
            return 0
        voxel_vol = get_aabb_volume(grid.aabb_from_voxel(next(iter(voxels))))
        return voxel_vol*len(voxels)



    def a_star(self, q_start, q_goal, v_0, H, relaxed):
        extended = set()
        paths = [([q_start], 0, 0)]

        while paths:
            print(paths)
            wait_if_gui()
            current = paths.pop(0)
            best_path = current[0]
            best_path_cost = current[1]
            best_path_h = current[2]

            if best_path[-1] in extended:
                continue

            if best_path[-1] == q_goal:
                return best_path

            extended.add(best_path[-1])
            for action in self.action_cost_fn(best_path[-1], v_0, relaxed):
                paths.append((best_path + [action[0]], best_path_cost+ action[1], H(action[0])))

            paths = sorted(paths, key=lambda x: x[-2] + x[-1])
        return None


    def execute_path(self, path):
        for qi, q in enumerate(path):
            set_joint_positions(self.env.robot, self.joints, q)

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(image_data)
            self.env.update_movable_boxes(image_data)
            self.env.update_visibility(camera_pose, image_data, q)

            # Check if remaining path is collision free under the new occupancy grid
            if len(self.obstruction_from_path(path[qi:])) != 0:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                return q, False
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)
        return q, True


def distance(vex1, vex2):
    dist = 0
    for i in range(len(vex1)):
        dist += (vex1[i] - vex2[i])**2
    return dist**0.5

