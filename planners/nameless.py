from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, AABB, OOBB, Pose, draw_oobb, LockRenderer,
                                                    Point, draw_aabb, set_joint_positions, joint_from_name,
                                                    get_link_pose, link_from_name, get_camera_matrix, draw_pose,
                                                    multiply, tform_point, invert, pixel_from_point, get_aabb_volume,
                                                    get_aabb_vertices, aabb_overlap, RED, BLACK)
import numpy as np
import time
import datetime
import scipy.spatial
import pickle
#import pybullet as p

from utils.graph import Graph
from environments.vamp_environment import GRID_RESOLUTION, find_min_angle

USE_COST = False


class Nameless(Planner):
    def __init__(self, env):
        # Sets up the environment and necessary data structures for the planner
        super(Nameless, self).__init__()

        self.env = env
        self.env.setup()

        # Initializes a graph that contains the available movements
        self.G = Graph()
        self.G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi/8])

        # Creates a voxel structure that contains the vision space
        self.env.setup_default_vision(self.G)

        # Specific joints to move the robot in simulation
        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

        # Structure used to save voxels that cannot be accessed by the robot, hence occupied
        self.occupied_voxels = dict()
        self.v_0 = None
        self.R = None
        self.complete = None
        self.current_q = None
        self.collided_object = None
        self.vision_q = dict()


    def get_plan(self, loadfile=None):
        """
        Creates a plan and executes it based on the given planner and environment.

        Args:
            loadfile (str): Location of the save file to load containing a previous state.
        Returns:
            list: The plan followed by the robot from start to goal.
        """
        q_start, q_goal = self.env.start, self.env.goal
        # Gets initial vision and updates the current vision based on it
        self.v_0 = self.get_circular_vision(q_start)
        self.env.update_vision_from_voxels(self.v_0)

        # Gathers vision from the robot's starting position and updates the
        # visibility and occupancy grids. Visualize them for convenience.
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data, q_start)
        self.env.update_occupancy(q_start, image_data)
        self.env.plot_grids(True, True, True)

        self.complete = False
        self.current_q = q_start

        # In case a loadfile is given. Load the state of the program to the specified one.
        if loadfile is not None:
            self.load_state("saves/" + loadfile)

            self.env.plot_grids(True, True, True)
            set_joint_positions(self.env.robot, self.joints, self.current_q)
            print("State loaded")
            print(self.collided_object)
            wait_if_gui()

        # Continue looking for a plan until the robot has reached the goal.
        while not self.complete:
            path = self.vamp_backchain(self.current_q, q_goal, self.v_0)
            # If at any point there is no possible path, then the search is ended.
            if path is None:
                print("Can't find path")
                break
            print("Found path:")
            print(path)

            # Execute path until it fails due to obstruction, or it reaches the goal. Update the
            # visibility based on what was observed while traversing the path.
            self.current_q, self.complete, gained_vision, self.collided_object = self.execute_path(path)
            self.v_0.update(gained_vision)

            # Ask for whether the user wants to save the current state to load it in the future.
            # print("Want to save this state? Press Y or N then Enter")
            # x = input()
            # if x == "Y" or x == "y":
            self.save_state()

        print("Reached the goal")
        wait_if_gui()

    def vamp_attach(self, q_start, q_goal, v_0):
        """
        Path planning using vamp for clearing an object out of the way.

        Args:
            q_start (tuple): Initial position from where to start to plan.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set): Set of voxels which indicate what areas of the world have been seen before
                    initializing the planning
        Returns:
            Whether we were able to attach successfully and the path through the object.
        """

        self.current_q = q_start
        complete = False
        p_obj = None

        while not complete:
            p = []
            v = v_0

            p_obj = self.vamp_path_vis(self.current_q, q_goal, v, relaxed=True, ignore_movable=True)
            obj_obstruction = self.env.find_path_movable_obstruction(p_obj)
            # Sample attachment poses
            attachment_poses = self.env.sample_attachment_poses(obj_obstruction, self.G)

            executed = False
            for pose in attachment_poses:
                path_exists = True
                while path_exists:
                    # Update vision since we might have moved in the process.
                    v.update(v_0)
                    # Get a direct path to the attachment
                    p_att = self.vamp_path_vis(self.current_q, pose, v)
                    if p_att is not None:
                        # If the path was found, executed until complete or a new obstacle was detected
                        p += p_att
                        self.current_q, complete, gained_vision, self.collided_object = self.execute_path(p)
                        self.v_0.update(gained_vision)
                        executed = True
                        break
                    # If no direct path exists find a relaxed one.
                    p_att_rel = self.vamp_path_vis(self.current_q, pose, v, relaxed=True)
                    if p_att_rel is None:
                        # If no relaxed path exists either, can continue to the next attachment pose
                        path_exists = False
                        continue
                    # View the required region if a relaxed path exists.
                    p_vis = self.vavp(self.current_q, self.env.visibility_voxels_from_path(p_att_rel).difference(v), v)
                    if p_vis is not None:
                        # Make this path a required action and try again on the same attachment pose.
                        p += p_vis
                        v = v.union(self.env.get_optimistic_path_vision(p_vis, self.G))
                        self.current_q = p_vis[-1]
                        continue
                    # Couldn't view the required area then try a new node
                    path_exists = False
                # If the robot moved at some point, we either reached the goal or found an obstacle.
                # Re-plan new attachments.
                if executed:
                    break

            # If we tried all the different attachments and couldn't find a path to any, explore
            # the environment and retry.
            if not executed and not complete:
                W = set(self.env.static_vis_grid.value_from_voxel.keys())
                p_vis = self.tourist(self.current_q, W.difference(v), v)
                if p_vis is None:
                    print("Can't do anything aborting")
                    return None, None

                self.current_q, _, gained_vision, self.collided_object = self.execute_path(p_vis)
                self.v_0.update(gained_vision)

        print("Attached successfully")
        # Ask for whether the user wants to save the current state to load it in the future.
        print("Want to save this state? Press Y or N then Enter")
        x = input()
        if x == "Y" or x == "y":
            self.save_state()

        return True, p_obj

    def vamp_place(self, q_start, p_through, v_0):
        """
        Path planning using vamp for clearing an object out of the way.

        Args:
            q_start (tuple): Initial position from where to start to plan.
            p_through (list): The path the object can follow to reach the goal, thus requiring
                                to leave it clear.
            v_0 (set): Set of voxels which indicate what areas of the world have been seen before
                    initializing the planning.
        Returns:
            Whether we were able to move the object successfully.
        """
        p = []
        v = v_0
        q = q_start
        grasp = None
        coll_obj = self.env.find_path_movable_obstruction(p_through)

        print("Ready to plan placement")
        self.env.remove_movable_object(coll_obj)
        executed = False
        while not executed:
            q_place, grasp, obj = self.env.sample_placement(q, coll_obj, self.G, p_through)
            self.env.movable_object_oobb_from_q(coll_obj, q_place, grasp, visualize=True)
            # Look for a path to move the object
            path_exists = True
            while path_exists:
                # Update vision since we might have moved in the process.
                v.update(v_0)
                p_att = self.vamp_path_vis(q, q_place, v, attachment=[coll_obj, grasp, obj])
                if p_att is not None:
                    # If the path was found, executed until complete or a new obstacle was detected
                    p += p_att
                    self.current_q, complete, gained_vision, self.collided_object = self.execute_path(p, attachment=[coll_obj, grasp, obj])
                    self.v_0.update(gained_vision)

                    # We executed so update variables
                    q = self.current_q
                    coll_obj = self.env.movable_object_oobb_from_q(coll_obj, q, grasp)
                    p = []

                    # If the object is still in the path we have to clear, then continue looking
                    # for a placement
                    obstruction = self.env.visibility_voxels_from_path(p_through)
                    if len(self.env.obstruction_from_path([q], obstruction,
                                                          attachment=[coll_obj, grasp])[0]) != 0:
                        break
                    # If the object is outside the path, then finish handling it.
                    executed = True
                    break

                # If it can't find direct path, find a relaxed one.
                p_att_rel = self.vamp_path_vis(q, q_place, v, relaxed=True,
                                               attachment=[coll_obj, grasp, obj])
                if p_att_rel is None:
                    # If no relaxed path exists either, can continue to the next attachment pose
                    path_exists = False
                    continue
                # View the required region if a relaxed path exists while restricting the path
                # through the object we want to move.
                obstructions = self.env.visibility_voxels_from_path(p_att_rel).difference(v)
                obstructions.update(self.env.occupancy_grid.voxels_from_aabb(coll_obj.aabb))
                p_vis = self.vavp(q, obstructions, v)
                if p_vis is not None:
                    # Make this path a required action and try again on the same attachment pose.
                    p += p_vis
                    v = v.union(self.env.get_optimistic_path_vision(p_vis, self.G))
                    self.current_q = p_vis[-1]
                path_exists = False

        # Place the object in the environment.
        self.env.movable_boxes.append(self.env.movable_object_oobb_from_q(coll_obj, self.current_q, grasp))
        print("Finished moving the object. Replanning")
        # Ask for whether the user wants to save the current state to load it in the future.
        print("Want to save this state? Press Y or N then Enter")
        x = input()
        if x == "Y" or x == "y":
            self.save_state()

        return True


    def vamp_backchain(self, q_start, q_goal, v_0):
        """
        Main function for path planning using VAMP.

        Args:
            q_start (tuple): Initial position from where to start to plan.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set): Set of voxels which indicate what areas of the world have been seen before
                    initializing the planning
        Returns:
            list: A suggested plan for the robot to follow.
        """
        p = []
        v = v_0
        q = q_start

        #success, p_through = self.vamp_attach(q_start, q_goal, v)
        #self.vamp_place(self.current_q, p_through, v)
        #p_final = self.vamp_path_vis(self.current_q, q_goal, v)

        while True:
            # Update vision since we might have moved in the process.
            v.update(v_0)
            # Find a path to goal, keeping the visualization constraint and return it if found
            p_final = self.vamp_path_vis(q, q_goal, v)
            if p_final is not None:
                return p + p_final
            print("Couldn't find a direct path. Looking for a relaxed one")

            # If a path to goal can't be found, find a relaxed path and use it as a subgoal
            p_relaxed = self.vamp_path_vis(q, q_goal, v, relaxed=True)
            if p_relaxed is None:
                print("Can't find any path. Looking through movable obstacles")
                attached_success, p_through = self.vamp_attach(q, q_goal, v)
                # If it can't find an attachment then no valid attachment exists. Break!
                if not attached_success:
                    return None
                q = self.current_q
                placed_success = self.vamp_place(q, p_through, v)
                q = self.current_q
                # TODO:Currently it never fails because we have infinite sampling. Change later.
                if not placed_success:
                    return None
                continue

            p_vis = self.vavp(q, self.env.visibility_voxels_from_path(p_relaxed).difference(v), v)
            # If the relaxed version fails, explore some environment. And restart the search
            if p_vis is None:
                print("P_VIS failed. Observing some of the environment")
                W = set(self.env.static_vis_grid.value_from_voxel.keys())
                p_vis = self.tourist(q, W.difference(v), v)
            if p_vis is None:
                print("P_VIS failed again. Aborting")
                return None

            p += p_vis
            v = v.union(self.env.get_optimistic_path_vision(p_vis, self.G))
            q = p_vis[-1]

    def vavp(self, q, R, v, obstructions=set()):
        """
        Subprocedure to aid the planning on dividing the objective into subgoals and planning paths
        accordingly.

        Args:
            q (tuple): Current position of the robot.
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            v (set) : Set of tuples that define the already seen space.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
        Returns:
            list: The suggested path that views some of the area of interest.
        """
        # Try visualizing the area of interest keeping the vision constraint.
        p_vis = self.tourist(q, R, v)
        if p_vis is not None:
            return p_vis
        # If it can't view the area, find a relaxed path that does the same and make this new path
        # the new subgoal. Call the function recursively.
        obstructions_new = obstructions.union(R)
        p_relaxed = self.tourist(q, R, v, relaxed=True, obstructions=obstructions_new)
        if p_relaxed is not None:
            p_vis = self.vavp(q, self.env.visibility_voxels_from_path(p_relaxed).difference(v), v, obstructions=obstructions_new)
            if p_vis is not None:
                return p_vis
        return None

    def tourist(self, q_start, R, v_0, relaxed=False, obstructions=set(), ignore_movable=False):
        """
        Procedure used to find a path that partially or completely views some area of interest.

        Args:
            q_start (tuple): Starting position of the robot.
            R (set) : Set of voxels that define the area we are interested in gaining vision from.
            v_0 (set) : Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
        Returns:
            list: The suggested path that views some area of interest.
        """
        q_goal = None
        score = 0
        number_of_samples = 1000
        # Sample a goal position that views most of the space of interest.
        for i in range(number_of_samples):
            q_rand = self.G.rand_vex()
            # Check collisions with obstacle and movable objects if required
            collisions, coll_objects = self.env.obstruction_from_path([q_rand], obstructions)
            if not collisions.shape[0] > 0 and (ignore_movable or coll_objects is None):
                new_score = len(self.env.get_optimistic_vision(q_rand, self.G).intersection(R))
                if new_score != 0:
                    if new_score > score:
                        q_goal = q_rand
                        score = new_score

        # Defines a heuristic function to use on the A* star search. Currently using the distance to goal.
        def heuristic_fn(q):
            return distance(q, q_goal)

            # Previously used code that defines the heuristic as the smallest distance from the vision
            # gained in the configuration to the area of interest.
            vision_q = self.env.get_optimistic_vision(q, self.G)
            if len(R.intersection(vision_q)) != 0:
                return 0
            if len(vision_q) == 0:
                return np.inf
            s1 = np.array(list(vision_q))
            s2 = np.array(list(R))
            return scipy.spatial.distance.cdist(s1, s2).min()*GRID_RESOLUTION

        return self.vamp_path_vis(q_start, q_goal, v_0, H=heuristic_fn, relaxed=relaxed, obstructions=obstructions)

    # def vamp_step_vis(self, q_start, q_goal, v_0, H=0, relaxed=False, obstructions=set(),
    #                   ignore_movable=False, attachment=None):
    #     """
    #     Helper function to initialize the search. Uses the vision constraint on each node based
    #     on the vision gained from the previous step only.

    #     Args:
    #         q_start (tuple): Starting position of the robot.
    #         q_goal (tuple): Goal position where the planning is ended.
    #         v_0 (set) : Set of tuples that define the already seen space.
    #         H (function): Function defining the heuristic used during A* search.
    #         relaxed (bool): Defines whether the path can relax the vision constraint.
    #         obstructions (set): Set of tuples that define the space that the robot can't occupy.
    #         ignore_movable (bool): Whether to ignore collisions with movable objects or not.
    #         attachment (list): A list of an attached object's oobb and its attachment grasp.
    #     Returns:
    #         list: The suggested path that goes from start to goal.
    #     """
    #     if H == 0:
    #         H = lambda x: distance(x, q_goal)

    #     return self.a_star(q_start, q_goal, v_0, H, relaxed, self.action_fn_step, obstructions=obstructions,
    #                        ignore_movable=ignore_movable, attachment=attachment)

    def vamp_path_vis(self, q_start, q_goal, v_0, H=0, relaxed=False, obstructions=set(),
                      ignore_movable=False, attachment=None):
        """
        Helper function to initialize the search. Uses the vision constraint on each node based
        on the vision gained from the first path found to the node.

        Args:
            q_start (tuple): Starting position of the robot.
            q_goal (tuple): Goal position where the planning is ended.
            v_0 (set): Set of tuples that define the already seen space.
            H (function): Function defining the heuristic used during A* search.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            list: The suggested path that goes from start to goal.
        """
        if H == 0:
            H = lambda x: distance(x, q_goal)

        return self.a_star(q_start, q_goal, v_0, H, relaxed, self.action_fn_path, obstructions=obstructions,
                           ignore_movable=ignore_movable, attachment=attachment)

    # def action_fn_step(self, path, v_0, relaxed=False, extended=set(), obstructions=set(),
    #                    ignore_movable=False, attachment=None):
    #     """
    #     Helper function to the search, that given a node, it gives all the possible actions to take with
    #     the inquired cost of each. Uses the vision constraint on each node based
    #     on the vision gained from the previous step only.

    #     Args:
    #         path (list): The path obtained to reach the current node on the search.
    #         v_0 (set): Set of tuples that define the already seen space.
    #         relaxed (bool): Defines whether the path can relax the vision constraint.
    #         extended (set): Set of nodes that were already extended by the search.
    #         obstructions (set): Set of tuples that define the space that the robot can't occupy.
    #         ignore_movable (bool): Whether to ignore collisions with movable objects or not.
    #         attachment (list): A list of an attached object's oobb and its attachment grasp.
    #     Returns:
    #         list: A list of available actions with the respective costs.
    #     """
    #     actions = []
    #     q = path[-1]
    #     # Retrieve all the neighbors of the current node based on the graph of the space.
    #     for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
    #         q_prime = self.G.vertices[q_prime_i]
    #         # If the node has already been extended do not consider it.
    #         if q_prime in extended:
    #             continue
    #         if relaxed:
    #             # Check for whether the new node is in obstruction with any obstacle.
    #             collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], obstructions,
    #                                                                       ignore_movable=ignore_movable,
    #                                                                       attachment=attachment)
    #             if not collisions.shape[0] > 0 and (ignore_movable or coll_objects is None):
    #                 v_q = v_0.union(self.env.get_optimistic_vision(q, self.G, attachment=attachment))
    #                 s_q = self.env.visibility_voxels_from_path([q, q_prime], attachment=attachment)
    #                 # If the node follows the visibility constraint, add it normally.
    #                 if s_q.issubset(v_q):
    #                     actions.append((q_prime, distance(q, q_prime)))
    #                 # If it does not follow the visibility constraint, add it with a special cost.
    #                 else:
    #                     cost = distance(q, q_prime) *\
    #                             abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
    #                     actions.append((q_prime, cost))
    #         else:
    #             # In the not relaxed case only add nodes when the visibility constraint holds.
    #             collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], obstructions,
    #                                                                       ignore_movable=ignore_movable,
    #                                                                       attachment=attachment)
    #             if not collisions.shape[0] > 0 and (ignore_movable or coll_objects is None):
    #                 v_q = v_0.union(self.env.get_optimistic_vision(q, self.G, attachment=attachment))
    #                 s_q = self.env.visibility_voxels_from_path([q, q_prime], attachment=attachment)
    #                 if s_q.issubset(v_q):
    #                     actions.append((q_prime, distance(q, q_prime)))
    #     return actions

    def action_fn_path(self, path, v_0, relaxed=False, extended=set(), obstructions=set(),
                       ignore_movable=False, attachment=None):
        """
        Helper function to the search, that given a node, it gives all the possible actions to take with
        the inquired cost of each. Uses the vision constraint on each node based
        on the vision gained from the first path found to the node.

        Args:
            path (list): The path obtained to reach the current node on the search.
            v_0 (set): Set of tuples that define the already seen space.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            extended (set): Set of nodes that were already extended by the search.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            list: A list of available actions with the respective costs.
        """
        actions = []
        q = path[-1]
        # Retrieve all the neighbors of the current node based on the graph of the space.
        for q_prime_i in self.G.neighbors[self.G.vex2idx[q]]:
            q_prime = self.G.vertices[q_prime_i]
            # If the node has already been extended do not consider it.
            if q_prime in extended:
                continue
            if relaxed:
                # Check for whether the new node is in obstruction with any obstacle.
                collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], obstructions,
                                                                          ignore_movable=ignore_movable,
                                                                          attachment=attachment)
                if not collisions.shape[0] > 0 and (ignore_movable or coll_objects is None):
                    if len(path) == 1:
                        v_q = v_0.union(self.env.get_optimistic_vision(q, self.G, attachment=attachment))
                    else:
                        v_q = self.vision_q[path[-2]].union(self.env.get_optimistic_vision(q, self.G,
                                                                                           attachment=attachment))
                    self.vision_q[q] = v_q
                    s_q = self.env.visibility_voxels_from_path([q, q_prime], attachment=attachment)
                    # If the node follows the visibility constraint, add it normally.
                    if s_q.issubset(v_q):
                        actions.append((q_prime, distance(q, q_prime)))
                    # If it does not follow the visibility constraint, add it with a special cost.
                    else:
                        cost = distance(q, q_prime) *\
                                abs(self.volume_from_voxels(self.env.static_vis_grid, s_q.difference(v_q)))
                        actions.append((q_prime, cost))
            else:
                # In the not relaxed case only add nodes when the visibility constraint holds.
                collisions, coll_objects = self.env.obstruction_from_path([q, q_prime], obstructions,
                                                                          ignore_movable=ignore_movable,
                                                                          attachment=attachment)
                if not collisions.shape[0] > 0 and (ignore_movable or coll_objects is None):
                    s_q = self.env.visibility_points_from_path([q, q_prime])
                    if self.env.in_view_cone(s_q, path):
                        actions.append((q_prime, distance(q, q_prime)))

        return actions


    def get_circular_vision(self, q, radius=1):
        """
        Gets a set of voxels that form a circle around a given point.

        Args:
            q (tuple): Center of the circle.
            radius (float): Radius of the circle
        Returns:
            set: A set of voxels representing a circular area around the center with the given radius.
        """
        grid = self.env.static_vis_grid
        surface_aabb = grid.aabb
        voxels = set()
        for voxel in grid.voxels_from_aabb(surface_aabb):
            actual_q = (q[0], q[1], 0)
            actual_vox = np.array(voxel) * np.array(self.G.res)
            if distance(actual_vox, actual_q) <= radius:
                voxels.add(voxel)
        return voxels


    def volume_from_voxels(self, grid, voxels):
        """
        Calculates the volume of a given set of voxels.

        Args:
            grid (object): The grid to which the voxels belong to.
            voxels (set): The set of voxels from which to determine the volume
        Returns:
            float: The volume of the voxels.
        """
        if len(voxels) == 0:
            return 0
        voxel_vol = get_aabb_volume(grid.aabb_from_voxel(next(iter(voxels))))
        return voxel_vol*len(voxels)

    def a_star(self, q_start, q_goal, v_0, H, relaxed, action_fn, obstructions=set(),
               ignore_movable=False, attachment=None):
        """
        A* search algorithm.

        Args:
            q_start (tuple): Start node.
            q_goal (tuple): Goal node.
            v_0 (set): Set of tuples that define the already seen space.
            H (function): Function defining the heuristic used during search.
            relaxed (bool): Defines whether the path can relax the vision constraint.
            action_fn (function): Defines the possible actions that any node can take and their costs.
            obstructions (set): Set of tuples that define the space that the robot can't occupy.
            ignore_movable (bool): Whether to ignore collisions with movable objects or not.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            list: The path from start to goal.
        """
        # Timing the search for benchmarking purposes.
        current_t = time.clock_gettime_ns(0)
        extended = set()
        paths = [([q_start], 0, 0)]

        while paths:
            current = paths.pop(-1)
            best_path = current[0]
            best_path_cost = current[1]

            # Ignore a node that has already been extended.
            if best_path[-1] in extended:
                continue

            # If goal is found return it, graph the search, and output the elapsed time.
            if best_path[-1] == q_goal:
                done = time.clock_gettime_ns(0) - current_t
                print(done * (10 ** (-9)))
                self.G.plot_search(self.env, extended, path=best_path, goal=q_goal)
                return best_path

            extended.add(best_path[-1])
            actions = action_fn(best_path, v_0, relaxed=relaxed, extended=extended,
                                obstructions=obstructions, ignore_movable=ignore_movable,
                                attachment=attachment)
            for action in actions:
                paths.append((best_path + [action[0]], best_path_cost + action[1], H(action[0])))

            # Only sorting from heuristic. Faster but change if needed
            if USE_COST:
                paths = sorted(paths, key=lambda x: x[-1], reverse=True)
            else:
                paths = sorted(paths, key=lambda x: x[-1], reverse=True)

        done = time.clock_gettime_ns(0) - current_t
        print(done * (10 ** (-9)))
        self.G.plot_search(self.env, extended, goal=q_goal)
        return None

    def execute_path(self, path, attachment=None):
        """
        Executes a given path in simulation until it is complete or no longer feasible.

        Args:
            path (list): The path to execute.
            attachment (list): A list of an attached object's oobb and its attachment grasp.
        Returns:
            tuple: A tuple containing the state where execution stopped, whether it was able to reach the goal,
             the gained vision, and the collided movable object.
        """
        gained_vision = set()
        for qi, q in enumerate(path):
            # Check whether the next step goes into area that is unseen.
            # next_occupied = self.env.visibility_points_from_path([q], attachment=attachment)
            # for voxel in next_occupied:
            #     if self.env.visibility_grid.contains(voxel):
            #         qi = qi-1 if qi-1 >= 0 else 0
            #         print("Stepping into unseen area. Aborting")
            #         return path[qi], False, gained_vision, None

            self.env.move_robot(q, self.joints, attachment=attachment)

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(q, image_data)
            gained_vision.update(self.env.update_movable_boxes(image_data))
            gained_vision.update(self.env.update_visibility(camera_pose, image_data, q))

            # If an object is attached, do not detect it as an obstacle or a new movable object
            if attachment is not None:
                self.env.clear_noise_from_attached(q, attachment)

            # Check if remaining path is collision free under the new occupancy grid
            obstructions, collided_obj = self.env.obstruction_from_path(path[qi:], set(), attachment=attachment)
            if len(obstructions) != 0 or collided_obj is not None:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                return q, False, gained_vision, collided_obj
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)

        return q, True, gained_vision, None


    def save_state(self):
        """
        Saves the current state of the Vamp planning algorithm.
        """
        current_time = datetime.datetime.now()
        dbfile = open("saves/state_{}_{}_{}_{}_{}.dat".format(current_time.month, current_time.day, current_time.hour,
                                           current_time.minute, current_time.second), "wb")
        pickle.dump(self, dbfile)
        dbfile.close()

    def load_state(self, filename):
        """
        Loads the specified file containing a state of the Vamp planner.

        Args:
            filename (str): The path to the file to load.
        """
        dbfile = open(filename, 'rb')
        copy = pickle.load(dbfile)

        self.env = copy.env
        self.G = copy.G
        self.occupied_voxels = copy.occupied_voxels
        self.v_0 = copy.v_0
        self.R = copy.R
        self.complete = copy.complete
        self.current_q = copy.current_q
        self.collided_object = copy.collided_object
        self.vision_q = copy.vision_q

        dbfile.close()


def distance(vex1, vex2):
    """
    Helper function that returns the Euclidean distance between two configurations.
    It uses a "fudge" factor for the relationship between angles and distances.

    Args:
        vex1 (tuple): The first tuple
        vex2 (tuple): The second tuple
    Returns:
        float: The Euclidean distance between both tuples.
    """
    R = 0.01
    dist = 0
    for i in range(len(vex1)-1):
        dist += (vex1[i] - vex2[i])**2
    dist += (R*find_min_angle(vex1[2], vex2[2]))**2
    return dist**0.5

