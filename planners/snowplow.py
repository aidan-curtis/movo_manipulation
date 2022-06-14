from planners.rrt import RRT
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions,
                                                    invert, multiply, pairwise_collisions, Pose, Point, Euler,
                                                    point_from_pose, sample_directed_reachable_base, get_pose,
                                                    get_aabb_center, set_pose, remove_all_debug, LockRenderer,
                                                    get_aabb, recenter_oobb, AABB, OOBB, aabb_union, draw_aabb,
                                                    RED, YELLOW, get_all_links)
import numpy as np
import pybullet as p
import time


class Snowplow(RRT):
    def __init__(self, env):
        super(Snowplow, self).__init__(env)
        self.env = env
        # Setup the environment (Not needed because of RRT already setting it up)
        #self.env.setup()

        self.step_size = [0.05, np.pi/18]
        self.RRT_ITERS = 5000
        self.COLLISION_DISTANCE = 5e-3

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

        self.movable_handles = []

    def find_path_obstruction(self, path):
        """
            Find the first movable object bounding box that this path collides with
        """
        for q in path:
            collision = self.env.check_conf_collision(q)
            if collision != False and collision != True:
                return collision

    def directed_pose_generator(self, pose, **kwargs):
        point = point_from_pose(pose)
        while True:
            base_values = sample_directed_reachable_base(self, point, **kwargs)
            if base_values is None:
                break
            yield tuple(list(base_values)+[0.1])


    def base_sample_gen(self, pose):
        return self.directed_pose_generator(pose, reachable_range=(0.8, 0.8))


    def sample_attachment_base_confs(self, robot, movable_box):
        pose = Pose(point=get_aabb_center(movable_box.aabb))
        base_generator = self.base_sample_gen(pose)
        for base_conf in base_generator:
            base_conf = base_conf[0:3]
            return base_conf


    def sample_attachment(self, movable_box):
        base_conf = self.sample_attachment_base_confs(self.env.robot, movable_box)
        base_pose = Pose(
            point=Point(x=base_conf[0], y=base_conf[1]),
            euler=Euler(yaw=base_conf[2]),
        )
        obj_pose = Pose(point=get_aabb_center(movable_box.aabb))
        base_grasp = multiply(invert(base_pose), obj_pose)

        return base_conf, base_grasp

    def sample_placement_poses(self, movable_box, grasp):
        invalid = True
        q = None
        while invalid:
            q = self.sample_from_vision()
            invalid = self.env.check_conf_collision_w_attached(q, movable_box, grasp)
        return q


    def get_detachment_path(self, movable_box, q, grasp):
        detachment_path = None
        while (detachment_path is None):
            detachment_q = self.sample_placement_poses(movable_box, grasp)
            detachment_path = self.get_path(q, detachment_q, attached_object=(movable_box, grasp), moving_backwards=True)
        return detachment_path


    def get_attachment_path(self, movable_box, q):
        attachment_path = None
        while(attachment_path is None):
            attachment_q, attachment_grasp = self.sample_attachment(movable_box)
            if self.env.check_conf_collision(attachment_q):
                continue
            attachment_path = self.get_path(q, attachment_q)
        return attachment_path, attachment_grasp


    def execute_path_with_attached(self, path, movable_object_oobb, grasp, ignore_movable=False):
        movable_obj = self.env.get_object_from_oobb(movable_object_oobb)
        for qi, q in enumerate(path):
            self.env.move_robot_with_att_obj(movable_obj, q, grasp, self.joints)

            # Get updated occupancy grid at each step
            camera_pose, image_data = self.env.get_robot_vision()
            self.env.update_occupancy(image_data, ignore_obstacles=[movable_obj])
            self.env.update_movable_boxes(image_data, ignore_obstacles=[movable_obj])
            self.env.update_visibility(camera_pose, image_data)

            # Check if remaining path is collision free under the new occupancy grid
            for next_qi in path[qi:]:
                if self.env.check_conf_collision_w_attached(q, movable_object_oobb, grasp):
                    self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                    return q, False
        return q, True


    def get_plan(self):
        
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data)
        self.env.update_occupancy(image_data)

        self.env.plot_grids(visibility=True, occupancy=True, movable=True)
        current_q, complete = self.env.start, False

        while(not complete):
            final_path = self.get_path(current_q, self.env.goal)
            if(final_path is None):
                print("No direct path to goal")
                relaxed_final_path = self.get_path(current_q, self.env.goal, ignore_movable=True)
                if(relaxed_final_path is None):
                    print("No indirect path to goal :(")
                else:
                    handling_complete = False
                    while not handling_complete:
                        obstruction = self.find_path_obstruction(relaxed_final_path)
                        obstructing_object = self.env.get_object_from_oobb(obstruction)
                        print("Found path through obstacle: " + str(obstruction))
                        attach_grasp = None
                        while (not handling_complete):
                            obstruction = self.find_path_obstruction(relaxed_final_path)
                            attach_path, attach_grasp = self.get_attachment_path(obstruction, current_q)
                            print("Found an attachment path")
                            current_q, handling_complete = self.execute_path(attach_path)
                            self.env.plot_grids(visibility=True, occupancy=True, movable=True)
                        print("Successfully attached to the object")

                        handling_complete = False
                        while (not handling_complete):
                            is_relaxed = False
                            obstruction = self.find_path_obstruction(relaxed_final_path)
                            print("Finding placement path")
                            #self.env.visualize_attachment_bbs(obstruction, current_q, attach_grasp)
                            self.env.remove_movable_object(obstruction)
                            detach_path = self.get_detachment_path(obstruction, current_q, attach_grasp)
                            print("Found placement path. Looking if we can reach goal from there")
                            newly_added = self.env.place_movable_object(obstruction, detach_path[-1], attach_grasp)
                            remaining_path = self.get_path(detach_path[-1], self.env.goal)
                            if remaining_path is None:
                                print("No direct path to goal. Looking for path through movable object")
                                remaining_path = self.get_path(detach_path[-1], self.env.goal, ignore_movable=True, forced_obj_coll=[newly_added])
                                self.env.remove_movable_object(newly_added)
                                is_relaxed = True
                                if remaining_path is None:
                                    print("Failed to retrieve path to goal, trying again")
                                    self.env.movable_boxes.append(obstruction)
                                    continue
                            self.env.remove_movable_object(newly_added)
                            print("We got it")
                            current_q, handling_complete = self.execute_path_with_attached(detach_path, obstruction, attach_grasp)
                            self.env.place_movable_object(obstruction, current_q, attach_grasp)
                            self.env.plot_grids(occupancy=True, visibility=True, movable=True)
                        if is_relaxed:
                            handling_complete = False
                            relaxed_final_path = remaining_path
            else:
                current_q, complete = self.execute_path(final_path)

        self.env.plot_grids(occupancy=True, visibility=True, movable=True)
        print("Reached the goal")
        wait_if_gui()
    
