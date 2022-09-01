import pickle
import datetime

from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import (joint_from_name, draw_aabb, wait_if_gui, draw_oobb, draw_pose, Pose,
                                                    Point, Euler, get_aabb_center, multiply, invert,
                                                    set_joint_positions, set_pose, get_pose)
import numpy as np
import time
from utils.graph import Graph
from environments.environment import GRID_RESOLUTION, find_min_angle
from itertools import groupby


class Snowplow(Planner):

    def __init__(self, env):
        self.env = env
        self.env.setup()

        # Initializes a graph that contains the available movements
        self.G = Graph()
        self.G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi / 8])

        # In case there is an environment with restricted configurations
        self.env.restrict_configuration(self.G)

        # Creates a voxel structure that contains the vision space
        self.env.setup_default_vision(self.G)

        # Specific joints to move the robot in simulation
        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

        # Structure used to save voxels that cannot be accessed by the robot, hence occupied
        self.occupied_voxels = dict()
        self.debug = False
        self.v_0 = None
        self.current_q = None
        self.final_executed = []
        self.object_poses = None



    def get_plan(self, debug=False, loadfile=None, **kwargs):
        self.debug = debug
        self.current_q, q_goal = self.env.start, self.env.goal
        # Gets initial vision and updates the current vision based on it
        self.v_0 = self.env.get_circular_vision(self.current_q, self.G)
        self.env.update_vision_from_voxels(self.v_0)

        # Gathers vision from the robot's starting position and updates the
        # visibility and occupancy grids. Visualize them for convenience.
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data, self.current_q)
        self.env.update_occupancy(self.current_q, image_data)
        self.env.update_movable_boxes(image_data)
        self.env.plot_grids(True, True, True)

        # In case a loadfile is given. Load the state of the program to the specified one.
        if loadfile is not None:
            self.load_state(loadfile)
            self.env.plot_grids(True, True, True)
            set_joint_positions(self.env.robot, self.joints, self.current_q)
            for i, obj in enumerate(self.env.room.movable_obstacles):
                set_pose(obj, self.object_poses[i])
            print("State loaded")

        complete = False
        while not complete:
            path = self.snowplow(self.current_q, q_goal, self.v_0)
            if path is None:
                return [key for key, _group in groupby(self.final_executed)]

            self.current_q, complete, gained_vision, executed_path = self.execute_path(path)
            self.final_executed += executed_path
            self.v_0.update(gained_vision)
            if self.debug:
                print("Want to save this state? Press Y or N then Enter")
                x = input()
                if x == "Y" or x == "y":
                    self.object_poses = []
                    for obj in self.env.room.movable_obstacles:
                        self.object_poses.append(get_pose(obj))
                    self.save_state()

        # Search for repeated nodes in a sequence and filter them.
        return [key for key, _group in groupby(self.final_executed)]


    def execute_path(self, path):
        """
        Executes a given path in simulation until it is complete or no longer feasible.

        Args:
            path (list): The path to execute.
        Returns:
            tuple: A tuple containing the state where execution stopped, whether it was able to reach the goal,
             the gained vision, and the executed path.
        """
        gained_vision = set()
        executed = []
        current_grasp, coll_obj = None, None
        attachment = None
        for qi, node in enumerate(path):
            q, obj = node

            # Check if we are grasping a new object
            if attachment is None and obj is not None:
                coll_obj = self.env.get_movable_box_from_obj(obj)
                # Compute the grasp transform of the attachment.
                base_pose = Pose(point=Point(x=q[0], y=q[1]), euler=Euler(yaw=q[2]),)
                obj_pose = Pose(point=get_aabb_center(coll_obj.aabb))
                current_grasp = multiply(invert(base_pose), obj_pose)
                attachment = [coll_obj, current_grasp, obj]
                self.env.remove_movable_object(coll_obj)
            elif attachment is not None and obj is None:
                oobb = self.env.movable_object_oobb_from_q(attachment[0], path[qi-1][0], attachment[1])
                self.env.movable_boxes.append(oobb)
                attachment = None


            self.env.move_robot(q, self.joints, attachment=attachment)
            # Executed paths saved as a list of q and attachment.
            executed.append([q, attachment])

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(q, image_data)
            gained_vision.update(self.env.update_movable_boxes(image_data))
            gained_vision.update(self.env.update_visibility(camera_pose, image_data, q))

            # If an object is attached, do not detect it as an obstacle or a new movable object
            # TODO: Find a better method to clear the noise than the current one
            if attachment is not None:
                self.env.clear_noise_from_attached(q, attachment)

            # Check if remaining path is collision free under the new occupancy grid
            # TODO only checking for obstruction in the next step. Think about changing
            #obstructions, collided_obj = self.env.obstruction_from_path(path_filtered[qi:qi+2], set(), attachment=attachment)
            obstructions, collided_obj = self.find_obstruction_ahead(path[qi:], attachment)
            if obstructions.shape[0] > 0 or collided_obj is not None:
                print("Found a collision on this path. Aborting")
                self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                if attachment is not None:
                    oobb = self.env.movable_object_oobb_from_q(attachment[0], q, attachment[1])
                    self.env.movable_boxes.append(oobb)
                return q, False, gained_vision, executed
            self.env.plot_grids(visibility=True, occupancy=True, movable=True)

        return q, True, gained_vision, executed

    def find_obstruction_ahead(self, path, att):
        placed_back = []
        needed_back = []
        if att is not None:
            oobb = self.env.movable_object_oobb_from_q(att[0], path[0][0], att[1])
            self.env.movable_boxes.append(oobb)
            placed_back.append(oobb)

        attachment = None
        for qi, node in enumerate(path):
            q, obj = node
            # Check if we are grasping a new object
            if attachment is None and obj is not None:
                coll_obj = self.env.get_movable_box_from_obj(obj)
                # Compute the grasp transform of the attachment.
                base_pose = Pose(point=Point(x=q[0], y=q[1]), euler=Euler(yaw=q[2]), )
                obj_pose = Pose(point=get_aabb_center(coll_obj.aabb))
                current_grasp = multiply(invert(base_pose), obj_pose)
                attachment = [coll_obj, current_grasp, obj]
                self.env.remove_movable_object(coll_obj)
                needed_back.append(coll_obj)

            elif attachment is not None and obj is None:
                oobb = self.env.movable_object_oobb_from_q(attachment[0], path[qi - 1][0], attachment[1])
                self.env.movable_boxes.append(oobb)
                placed_back.append(oobb)
                attachment = None

            obstructions, collided_obj = self.env.obstruction_from_path([q], set(),
                                                                    attachment=attachment)
            if obstructions.shape[0] > 0 or collided_obj is not None:
                self.env.movable_boxes += needed_back
                for box in placed_back:
                    self.env.remove_movable_object(box)
                return obstructions, collided_obj
        self.env.movable_boxes += needed_back
        for box in placed_back:
            self.env.remove_movable_object(box)
        return obstructions, collided_obj




    def save_state(self):
        """
        Saves the current state of the Vamp planning algorithm.
        """
        current_time = datetime.datetime.now()
        dbfile = open("saves/{}_state_{}_{}_{}_{}_{}_{}.dat".format(self.env.__class__.__name__,
                                                                    self.__class__.__name__, current_time.month,
                                                                    current_time.day, current_time.hour,
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
        self.current_q = copy.current_q
        self.final_executed = copy.final_executed
        self.object_poses = copy.object_poses
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
    r = 0.01
    dist = 0
    for i in range(len(vex1)-1):
        dist += (vex1[i] - vex2[i])**2
    dist += (r*find_min_angle(vex1[2], vex2[2]))**2
    return dist**0.5














# from planners.rrt import RRT
# from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name,
#                                                     invert, multiply, Pose, Point, Euler,
#                                                     point_from_pose, sample_directed_reachable_base,
#                                                     get_aabb_center)
# import numpy as np
# import random
#
# class Snowplow(RRT):
#     def __init__(self, env):
#         super(Snowplow, self).__init__(env)
#         self.env = env
#         # Setup the environment (Not needed because of RRT already setting it up)
#         #self.env.setup()
#
#         #self.step_size = [0.05, np.pi/18]
#         self.RRT_ITERS = 5000
#         self.COLLISION_DISTANCE = 5e-3
#
#         self.joints = [joint_from_name(self.env.robot, "x"),
#                        joint_from_name(self.env.robot, "y"),
#                        joint_from_name(self.env.robot, "theta")]
#
#         self.movable_handles = []
#
#     def find_path_obstruction(self, path):
#         """
#             Find the first movable object bounding box that this path collides with
#         """
#         for q in path:
#             collision = self.env.check_conf_collision(q)
#             if collision != False and collision != True:
#                 return collision
#
#     def directed_pose_generator(self, pose, **kwargs):
#         point = point_from_pose(pose)
#         while True:
#             base_values = sample_directed_reachable_base(self, point, **kwargs)
#             if base_values is None:
#                 break
#             yield tuple(list(base_values)+[0.1])
#
#
#     def base_sample_gen(self, pose):
#         return self.directed_pose_generator(pose, reachable_range=(0.8, 0.8))
#
#
#     def sample_attachment_base_confs(self, robot, movable_box):
#         pose = Pose(point=get_aabb_center(movable_box.aabb))
#         base_generator = self.base_sample_gen(pose)
#         for base_conf in base_generator:
#             base_conf = base_conf[0:3]
#             return base_conf
#
#
#     def get_grasp(self, base_conf, movable_box):
#         base_pose = Pose(
#             point=Point(x=base_conf[0], y=base_conf[1]),
#             euler=Euler(yaw=base_conf[2]),
#         )
#         obj_pose = Pose(point=get_aabb_center(movable_box.aabb))
#         base_grasp = multiply(invert(base_pose), obj_pose)
#         return base_grasp
#
#     def sample_attachment(self, movable_box):
#         base_conf = self.sample_attachment_base_confs(self.env.robot, movable_box)
#         base_grasp = self.get_grasp(base_conf, movable_box)
#
#         return base_conf, base_grasp
#
#     def sample_placement_poses(self, movable_box, grasp):
#         invalid = True
#         q = None
#         while invalid:
#             q = self.sample_from_vision()
#             invalid = self.env.check_conf_collision_w_attached(q, movable_box, grasp)
#         return q
#
#
#     def get_detachment_path(self, movable_box, q, grasp):
#         detachment_path = None
#         while (detachment_path is None):
#             detachment_q = self.sample_placement_poses(movable_box, grasp)
#             detachment_path = self.get_path(q, detachment_q, attached_object=(movable_box, grasp), moving_backwards=True)
#         return detachment_path
#
#
#     def get_attachment_path(self, movable_box, q):
#         attachment_path = None
#         while(attachment_path is None):
#             attachment_q, attachment_grasp = self.sample_attachment(movable_box)
#             if self.env.check_conf_collision(attachment_q):
#                 continue
#             attachment_path = self.get_path(q, attachment_q)
#         return attachment_path, attachment_grasp
#
#
#     def execute_path_with_attached(self, path, movable_object_oobb, grasp, ignore_movable=False):
#         movable_obj = self.env.get_object_from_oobb(movable_object_oobb)
#         for qi, q in enumerate(path):
#             self.env.move_robot_with_att_obj(movable_obj, q, grasp, self.joints)
#
#             # Get updated occupancy grid at each step
#             camera_pose, image_data = self.env.get_robot_vision()
#             self.env.update_occupancy(image_data, ignore_obstacles=[movable_obj])
#             self.env.update_movable_boxes(image_data, ignore_obstacles=[movable_obj])
#             self.env.update_visibility(camera_pose, image_data)
#
#             # Check if remaining path is collision free under the new occupancy grid
#             for next_qi in path[qi:]:
#                 if self.env.check_conf_collision_w_attached(next_qi, movable_object_oobb, grasp):
#                     self.env.plot_grids(visibility=True, occupancy=True, movable=True)
#                     return q, False
#         return q, True
#
#
#     def sample_from_vision(self):
#         resolution = self.env.visibility_grid.resolutions
#         free_points = [(free[0]*resolution[0], free[1]*resolution[1])
#                        for free in self.env.viewed_voxels]
#         point = random.choice(free_points)
#         rand_t = np.random.uniform(0, 2 * np.pi)
#
#         return (point[0], point[1], rand_t)
#
#
#     def get_plan(self):
#
#         camera_pose, image_data = self.env.get_robot_vision()
#         self.env.update_visibility(camera_pose, image_data)
#         self.env.update_occupancy(image_data)
#
#         self.env.plot_grids(visibility=True, occupancy=True, movable=True)
#         current_q, complete = self.env.start, False
#
#         while(not complete):
#             final_path = self.get_path(current_q, self.env.goal)
#             if(final_path is None):
#                 print("No direct path to goal")
#                 relaxed_final_path = self.get_path(current_q, self.env.goal, ignore_movable=True)
#                 if(relaxed_final_path is None):
#                     print("No indirect path to goal :(")
#                 else:
#                     handling_complete = False
#                     while not handling_complete:
#                         obstruction = self.find_path_obstruction(relaxed_final_path)
#                         obstructing_object = self.env.get_object_from_oobb(obstruction)
#                         print("Found path through obstacle: " + str(obstruction))
#                         attach_grasp = None
#                         while (not handling_complete):
#                             obstruction = self.find_path_obstruction(relaxed_final_path)
#                             attach_path, attach_grasp = self.get_attachment_path(obstruction, current_q)
#                             print("Found an attachment path")
#                             current_q, handling_complete = self.execute_path(attach_path)
#                             self.env.plot_grids(visibility=True, occupancy=True, movable=True)
#                         print("Successfully attached to the object")
#
#                         handling_complete = False
#                         while (not handling_complete):
#                             is_relaxed = False
#                             obstruction = self.find_path_obstruction(relaxed_final_path)
#                             attach_grasp = self.get_grasp(current_q, obstruction)
#                             print("Finding placement path")
#                             #self.env.visualize_attachment_bbs(obstruction, current_q, attach_grasp)
#                             self.env.remove_movable_object(obstruction)
#                             print(obstruction)
#                             print(self.env.movable_boxes)
#                             detach_path = self.get_detachment_path(obstruction, current_q, attach_grasp)
#                             print("Found placement path. Looking if we can reach goal from there")
#                             newly_added = self.env.place_movable_object(obstruction, detach_path[-1], attach_grasp)
#                             remaining_path = self.get_path(detach_path[-1], self.env.goal)
#                             if remaining_path is None:
#                                 print("No direct path to goal. Looking for path through movable object")
#                                 remaining_path = self.get_path(detach_path[-1], self.env.goal, ignore_movable=True, forced_obj_coll=[newly_added])
#                                 self.env.remove_movable_object(newly_added)
#                                 is_relaxed = True
#                                 if remaining_path is None:
#                                     print("Failed to retrieve path to goal, trying again")
#                                     self.env.movable_boxes.append(obstruction)
#                                     continue
#                             self.env.remove_movable_object(newly_added)
#                             print("We got it")
#                             current_q, handling_complete = self.execute_path_with_attached(detach_path, obstruction, attach_grasp)
#                             self.env.place_movable_object(obstruction, current_q, attach_grasp)
#                             self.env.plot_grids(occupancy=True, visibility=True, movable=True)
#                         if is_relaxed:
#                             handling_complete = False
#                             relaxed_final_path = remaining_path
#             else:
#                 current_q, complete = self.execute_path(final_path)
#
#         self.env.plot_grids(occupancy=True, visibility=True, movable=True)
#         print("Reached the goal")
#         wait_if_gui()
#
