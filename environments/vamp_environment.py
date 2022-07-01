from abc import ABC, abstractmethod
from cProfile import label
from email.policy import default
from re import L
from pybullet_planning.pybullet_tools.utils import (LockRenderer, load_pybullet, set_joint_positions, joint_from_name,
                                                    Point, Pose, Euler,
                                                    set_pose, create_box, TAN, get_link_pose,
                                                    get_camera_matrix, get_image_at_pose, tform_point, invert, multiply,
                                                    pixel_from_point, AABB, OOBB, BLUE, RED, YELLOW, link_from_name,
                                                    aabb_contains_point,
                                                    get_aabb, RGBA, recenter_oobb, get_aabb, draw_oobb,
                                                    aabb_from_points, OOBB,
                                                    aabb_union, aabb_overlap, scale_aabb, get_aabb_center,
                                                    remove_handles, get_pose, draw_aabb, wait_if_gui)
from pybullet_planning.pybullet_tools.voxels import (VoxelGrid)
from utils.motion_planning_interface import DEFAULT_JOINTS
from utils.utils import iterate_point_cloud
import pybullet as p
import os
import numpy as np
from collections import namedtuple, defaultdict
from functools import cached_property

GRID_HEIGHT = 2  # Height of the visibility and occupancy grids
GRID_RESOLUTION = 0.1  # Grid resolutions

LIGHT_GREY = RGBA(0.7, 0.7, 0.7, 1)

Room = namedtuple("Room", ["walls", "floors", "aabb", "movable_obstacles"])
Force = namedtuple("Force", ["magnitude", "angle"])


class Environment(ABC):

    @abstractmethod
    def setup(self):
        pass

    def validate_plan(self, plan):
        """
            Validates that the plan is collision-free, scores the trajectory cost
        """
        stats = {"success": True}
        return stats

    def get_object_from_oobb(self, oobb):
        for object in self.objects:
            if aabb_contains_point(get_aabb_center(oobb.aabb), get_aabb(object)):
                return object
        return None

    def update_movable_boxes(self, camera_image, ignore_obstacles=[], **kwargs):
        relevant_cloud = [lp for lp in iterate_point_cloud(camera_image, **kwargs)
                          if aabb_contains_point(lp.point, self.room.aabb)
                          ]

        object_points = defaultdict(list)
        for labeled_point in relevant_cloud:
            if (labeled_point.label[0] in self.room.movable_obstacles and labeled_point.label[
                0] not in ignore_obstacles):
                object_points[labeled_point.label[0]].append(labeled_point.point)

        # Convert point clusters into bounding boxes
        new_boxes = []
        for _, points in object_points.items():
            new_boxes.append(OOBB(aabb_from_points(points), Pose()))

        # If a bounding box intersects with an existing one, replace with union
        def test_in(box, boxes):
            return any([(all(np.array(box.aabb.upper) == np.array(q.aabb.upper)) and
                         all(np.array(box.aabb.lower) == np.array(q.aabb.lower))) for q in boxes])

        all_new_boxes = []
        overlapped_boxes = []
        for movable_box in self.movable_boxes:
            for new_box in new_boxes:
                if (aabb_overlap(movable_box.aabb, new_box.aabb)):
                    all_new_boxes.append(OOBB(aabb_union([movable_box.aabb, new_box.aabb]), Pose()))
                    overlapped_boxes.append(movable_box)
                    overlapped_boxes.append(new_box)
        for b in new_boxes + self.movable_boxes:
            if not (test_in(b, overlapped_boxes)):
                all_new_boxes.append(b)

        # Remove points from occupancy/visibility grids
        for movable_box in all_new_boxes:
            for voxel in self.occupancy_grid.voxels_from_aabb(scale_aabb(movable_box.aabb, 1.2)):
                self.occupancy_grid.set_free(voxel)
                self.visibility_grid.set_free(voxel)

        self.movable_boxes = all_new_boxes

    def update_occupancy(self, camera_image, ignore_obstacles=[], **kwargs):
        relevant_cloud = [lp for lp in iterate_point_cloud(camera_image, **kwargs)
                          if aabb_contains_point(lp.point, self.room.aabb)
                          ]
        for labeled_point in relevant_cloud:
            if labeled_point.label[0] not in ignore_obstacles:
                point = labeled_point.point
                self.occupancy_grid.add_point(point)

    def update_visibility(self, camera_pose, camera_image):
        surface_aabb = self.visibility_grid.aabb
        camera_pose, camera_matrix = camera_image[-2:]
        grid = self.visibility_grid
        for voxel in grid.voxels_from_aabb(surface_aabb):
            center_world = grid.to_world(grid.center_from_voxel(voxel))
            center_camera = tform_point(invert(camera_pose), center_world)
            distance = center_camera[2]
            pixel = pixel_from_point(camera_matrix, center_camera)
            if pixel is not None:
                r, c = pixel
                depth = camera_image.depthPixels[r, c]
                if distance <= depth:
                    self.viewed_voxels.append(voxel)
                    grid.set_free(voxel)
        return grid

    def setup_grids(self):
        self.setup_occupancy_grid()
        self.setup_visibility_grid()
        self.setup_movable_boxes()

    def setup_visibility_grid(self):
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        # surface_aabb = self.room.aabb
        surface_aabb = AABB(lower=self.room.aabb.lower,
                            upper=(self.room.aabb.upper[0], self.room.aabb.upper[1], 0.1))
        grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=BLUE
        )
        for voxel in grid.voxels_from_aabb(surface_aabb):
            grid.set_occupied(voxel)

        self.visibility_grid = grid

    def setup_occupancy_grid(self):
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        surface_aabb = self.room.aabb
        grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=RED
        )
        self.occupancy_grid = grid

    def setup_movable_boxes(self):
        self.movable_boxes = []

    def set_defaults(self, robot):
        joints, values = zip(*[(joint_from_name(robot, k), v) for k, v in DEFAULT_JOINTS.items()])
        set_joint_positions(robot, joints, values)

    def setup_robot(self):
        MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
        MOVO_PATH = os.path.abspath(MOVO_URDF)
        robot_body = load_pybullet(MOVO_PATH, fixed_base=True)

        self.set_defaults(robot_body)
        return robot_body

    def plot_grids(self, visibility=False, occupancy=False, movable=False):
        movable_handles = []
        with LockRenderer():
            p.removeAllUserDebugItems()
            if (visibility):
                self.visibility_grid.draw_intervals()
            if (occupancy):
                self.occupancy_grid.draw_intervals()
            if (movable):
                for movable_box in self.movable_boxes:
                    draw_oobb(movable_box, color=YELLOW)
        return

    def get_robot_vision(self):
        """
        Gets the rgb and depth image of the robot
        """
        fx = 80
        fy = 80
        width = 128
        height = 128
        far = 100

        # 13 is the link of the optical frame of the rgb camera
        camera_link = link_from_name(self.robot, "kinect2_rgb_optical_frame")
        camera_pose = get_link_pose(self.robot, camera_link)

        camera_matrix = get_camera_matrix(width, height, fx, fy)
        camera_image = get_image_at_pose(camera_pose, camera_matrix, far=far, segment=True)

        return camera_pose, camera_image

    def create_closed_room(self, length, width, center=[0, 0], wall_height=2, movable_obstacles=[]):

        floor = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor, Pose(Point(x=center[0], y=center[1])))

        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1,
                 Pose(point=Point(x=center[0], y=center[1] + length / 2 + wall_thickness / 2, z=wall_height / 2)))
        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2,
                 Pose(point=Point(x=center[0], y=center[1] - (length / 2 + wall_thickness / 2), z=wall_height / 2)))
        wall_3 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3,
                 Pose(point=Point(y=center[1], x=center[0] + width / 2 + wall_thickness / 2, z=wall_height / 2)))
        wall_4 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4,
                 Pose(point=Point(y=center[1], x=center[0] - (width / 2 + wall_thickness / 2), z=wall_height / 2)))
        aabb = AABB(lower=(center[0] - width / 2.0, center[1] - length / 2.0, 0.05),
                    upper=(center[0] + width / 2.0, center[1] + length / 2.0, 0 + GRID_HEIGHT))
        return Room([wall_1, wall_2, wall_3, wall_4], [floor], aabb, movable_obstacles)

    def create_pillar(self, width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
        return create_box(w=width, l=length, h=height, color=color, **kwargs)

    def check_state_collision(self, joints, q):
        set_joint_positions(self.robot, joints, q)
        if self.occupancy_grid.get_affected([self.robot], True):
            return True

        # Check for the robot being inside the room. Disable later when replanning
        # with occupancy grid
        lower1, upper1 = get_aabb(self.robot)
        lower2, upper2 = self.room.aabb
        intersect = np.less_equal(lower2[0:2], lower1[0:2]).all() and np.less_equal(upper1[0:2], upper2[0:2]).all()
        if intersect == False:
            return True

        return False

    def check_conf_collision(self, q, ignore_movable=False, forced_object_coll=[]):
        aabb = self.centered_aabb
        z_centering = aabb[1][2]
        aabb = AABB(lower=[aabb[0][0] + q[0], aabb[0][1] + (q[1]), aabb[0][2] + z_centering],
                    upper=[aabb[1][0] + q[0], aabb[1][1] + (q[1]), aabb[1][2] + z_centering])

        if (not ignore_movable):
            for movable_box in self.movable_boxes:
                if (aabb_overlap(movable_box.aabb, aabb)):
                    return movable_box
        for movable_box in forced_object_coll:
            if (aabb_overlap(movable_box.aabb, aabb)):
                return movable_box
        for voxel in self.occupancy_grid.voxels_from_aabb(aabb):
            if self.occupancy_grid.is_occupied(voxel) == True:
                return True

        return False

    def check_conf_collision_w_attached(self, q, attached_object, grasp, ignore_movable=False):
        robot_pose = Pose(
            point=Point(x=q[0], y=q[1]),
            euler=Euler(yaw=q[2]),
        )
        obj_pose = multiply(robot_pose, grasp)
        aabb = self.centered_aabb
        midz = (aabb[1][2])
        robot_aabb = AABB(lower=[aabb[0][0] + q[0], aabb[0][1] + (q[1]), aabb[0][2] + midz],
                          upper=[aabb[1][0] + q[0], aabb[1][1] + (q[1]), aabb[1][2] + midz])

        aabb_object, _ = recenter_oobb((attached_object.aabb, obj_pose))
        object_aabb = AABB(lower=[aabb_object[0][0] + obj_pose[0][0], aabb_object[0][1] + obj_pose[0][1],
                                  aabb_object[0][2] + obj_pose[0][2]],
                           upper=[aabb_object[1][0] + obj_pose[0][0], aabb_object[1][1] + obj_pose[0][1],
                                  aabb_object[1][2] + obj_pose[0][2]])
        for aabb in [robot_aabb, object_aabb]:
            for voxel in self.occupancy_grid.voxels_from_aabb(aabb):
                if self.occupancy_grid.is_occupied(voxel):
                    return True
            for voxel in self.visibility_grid.voxels_from_aabb(aabb):
                if self.visibility_grid.is_occupied(voxel):
                    return True
            if not ignore_movable:
                for movable_box in self.movable_boxes:
                    if (aabb_overlap(movable_box.aabb, aabb)):
                        return movable_box
        return False

    def get_centered_aabb(self):
        # TODO: Using the base aabb for simplicity. Change later
        centered_aabb, _ = recenter_oobb((get_aabb(self.robot, link=4), Pose()))
        centered_aabb.lower[1] = centered_aabb.lower[0]
        centered_aabb.upper[1] = centered_aabb.upper[0]
        return centered_aabb

    def check_collision_in_path(self, q_init, q_final, resolution=0.1,
                                ignore_movable=False, forced_object_coll=[],
                                attached_object=None, moving_backwards=False):
        # qs = divide_path_on_resol(q_init, q_final, resolution)
        qs = [q_init, q_final]
        if moving_backwards:
            qs = self.adjust_angles_backwards(qs, q_init, q_final)
        else:
            qs = self.adjust_angles(qs, q_init, q_final)
        for q in qs:
            if attached_object is not None:
                if self.check_conf_collision_w_attached(q, attached_object[0], attached_object[1],
                                                        ignore_movable=False):
                    return True
            else:
                if self.check_conf_collision(q, ignore_movable=ignore_movable, forced_object_coll=forced_object_coll):
                    return True
        return False

    def move_robot_with_att_obj(self, movable_obj, q, attachment_grasp, joints):
        robot_pose = Pose(
            point=Point(x=q[0], y=q[1]),
            euler=Euler(yaw=q[2]),
        )
        obj_pose = multiply(robot_pose, attachment_grasp)
        set_joint_positions(self.robot, joints, q)
        set_pose(movable_obj, obj_pose)

    def adjust_angles_fast(self, path, start, goal):
        final_path = [start]
        for i in range(1, len(path)):
            beg = path[i - 1]
            end = path[i]

            delta_x = end[0] - beg[0]
            delta_y = end[1] - beg[1]
            theta = np.arctan2(delta_y, delta_x)

            final_path.append((beg[0], beg[1], theta))
            final_path.append((end[0], end[1], theta))
        final_path.append(goal)
        return final_path

    def adjust_angles(self, path, start, goal):
        final_path = [start]
        for i in range(1, len(path)):
            angle_traversal = np.pi / 12
            beg = path[i - 1]
            end = path[i]

            delta_x = end[0] - beg[0]
            delta_y = end[1] - beg[1]
            theta = np.arctan2(delta_y, delta_x)

            angle_diff = find_min_angle(final_path[-1][2], theta)
            n_iters = int(abs(angle_diff / angle_traversal))
            angle = final_path[-1][2]
            if angle_diff < 0: angle_traversal *= -1
            for e in range(n_iters):
                angle += angle_traversal
                final_path.append((beg[0], beg[1], angle))

            final_path.append((beg[0], beg[1], theta))
            final_path.append((end[0], end[1], theta))
        final_path.append(goal)
        return final_path

    def adjust_angles_backwards_old(self, path, start, goal):
        final_path = [goal]
        for i in range(len(path) - 1, 0, -1):
            beg = path[i]
            end = path[i - 1]

            delta_x = end[0] - beg[0]
            delta_y = end[1] - beg[1]
            theta = np.arctan2(delta_y, delta_x)

            final_path.append((beg[0], beg[1], theta))
            final_path.append((end[0], end[1], theta))
        final_path.append(start)
        final_path.reverse()
        return final_path

    def adjust_angles_backwards(self, path, start, goal):
        final_path = [goal]
        for i in range(len(path) - 1, 0, -1):
            angle_traversal = np.pi / 12
            beg = path[i]
            end = path[i - 1]

            delta_x = end[0] - beg[0]
            delta_y = end[1] - beg[1]
            theta = np.arctan2(delta_y, delta_x)

            angle_diff = find_min_angle(beg[2], theta)
            n_iters = int(abs(angle_diff / angle_traversal))
            angle = beg[2]
            if angle_diff < 0: angle_traversal *= -1
            for e in range(n_iters):
                angle += angle_traversal
                final_path.append((beg[0], beg[1], angle))

            final_path.append((beg[0], beg[1], theta))
            final_path.append((end[0], end[1], theta))
        final_path.append(start)
        final_path.reverse()
        return final_path

    def place_movable_object(self, movable_object_oobb, q, grasp):
        aabb = self.centered_aabb
        midz = (aabb[1][2])
        robot_aabb = AABB(lower=[aabb[0][0] + q[0], aabb[0][1] + (q[1]), aabb[0][2] + midz],
                          upper=[aabb[1][0] + q[0], aabb[1][1] + (q[1]), aabb[1][2] + midz])
        robot_pose = Pose(
            point=Point(x=q[0], y=q[1]),
            euler=Euler(yaw=q[2]),
        )
        obj_pose = multiply(robot_pose, grasp)
        aabb_object, _ = recenter_oobb((movable_object_oobb.aabb, obj_pose))
        object_aabb = AABB(
            lower=[aabb_object[0][0] + obj_pose[0][0], aabb_object[0][1] + obj_pose[0][1],
                   aabb_object[0][2] + obj_pose[0][2]],
            upper=[aabb_object[1][0] + obj_pose[0][0], aabb_object[1][1] + obj_pose[0][1],
                   aabb_object[1][2] + obj_pose[0][2]])
        # draw_aabb(robot_aabb)
        # draw_aabb(object_aabb)
        new_object = OOBB(aabb=object_aabb, pose=Pose())
        self.movable_boxes.append(new_object)
        return new_object

    def remove_movable_object(self, movable_object_oobb):
        remaining_boxes = []
        for i in range(len(self.movable_boxes)):
            good = True
            if (abs(np.array(self.movable_boxes[i].aabb.lower) - np.array(
                    movable_object_oobb.aabb.lower)) > 0.01).any():
                remaining_boxes.append(self.movable_boxes[i])
                good = False
                break
            if (abs(np.array(self.movable_boxes[i].aabb.upper) - np.array(
                    movable_object_oobb.aabb.upper)) > 0.01).any():
                remaining_boxes.append(self.movable_boxes[i])
                good = False
                break
        self.movable_boxes = remaining_boxes
        return False

    def visualize_attachment_bbs(self, movable_object_oobb, q, grasp):
        aabb = self.centered_aabb
        midz = aabb[1][2]
        robot_aabb = AABB(lower=[aabb[0][0] + q[0], aabb[0][1] + q[1], aabb[0][2] + midz],
                          upper=[aabb[1][0] + q[0], aabb[1][1] + q[1], aabb[1][2] + midz])
        robot_pose = Pose(
            point=Point(x=q[0], y=q[1]),
            euler=Euler(yaw=q[2]),
        )
        obj_pose = multiply(robot_pose, grasp)
        aabb_object, _ = recenter_oobb((movable_object_oobb.aabb, obj_pose))
        object_aabb = AABB(
            lower=[aabb_object[0][0] + obj_pose[0][0], aabb_object[0][1] + obj_pose[0][1],
                   aabb_object[0][2] + obj_pose[0][2]],
            upper=[aabb_object[1][0] + obj_pose[0][0], aabb_object[1][1] + obj_pose[0][1],
                   aabb_object[1][2] + obj_pose[0][2]])
        draw_aabb(robot_aabb)
        draw_aabb(object_aabb)
        return

    def push(self, force, position, object):
        friction_coeff = 0.1
        m = self.objects_prop[object][3]
        g = 9.8
        delta_t = 0.1
        I = 1 / 12 * m * (self.objects_prop[object][0] ** 2 + self.objects_prop[object][1] ** 2)
        object_center = get_pose(object)[0][:2]

        delta_x = position[0] - object_center[0]
        delta_y = position[1] - object_center[1]
        theta = np.arctan2(delta_y, delta_x)
        # print("Angle of the torque: {}".format(theta))

        rel_theta = force.angle - theta
        # print("Relative theta: {}".format(rel_theta))

        actual_force = force.magnitude - g * m * friction_coeff
        force_x = round(actual_force * np.cos(rel_theta), 3)
        force_y = round(actual_force * np.sin(rel_theta), 3)

        d = ((position[0] - object_center[0]) ** 2 + (position[1] - object_center[1]) ** 2) ** 0.5

        torque = d * force_y
        angular_dis = torque / I * (delta_t ** 2) / 2
        # print("Angular displacement: {}".format(angular_dis))

        s = force_x / m * (delta_t ** 2) / 2

        s_x = s * np.cos(theta)
        s_y = s * np.sin(theta)
        # print("Relative displacements: {}, {}".format(s_x, s_y))

        # print("Torque: {}".format(torque))
        # print("Relative forces: {} , {}".format(force_x, force_y))
        # print("Object center: {}".format(object_center))

        transform = Pose(point=Point(x=s_x, y=s_y, z=0), euler=(0, 0, angular_dis))
        new_pose = multiply(get_pose(object), transform)
        # print("New Pose: {}".format(new_pose))
        set_pose(object, new_pose)

        return new_pose


def divide_path_on_resol(q_init, q_final, step_size):
    dirn = np.array(q_final[0:2]) - np.array(q_init[0:2])
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(step_size, length)

    path = [q_init]
    i = 0
    while True:
        new_vex = (path[-1][0] + dirn[0], path[-1][1] + dirn[1], q_init[2])
        path.append(new_vex)
        if distance(new_vex, q_final) < step_size:
            path.append(q_final)
            return path


def distance(vex1, vex2):
    return ((vex1[0] - vex2[0]) ** 2 + (vex1[1] - vex2[1]) ** 2) ** 0.5


def find_min_angle(beg, end):
    if beg > np.pi:
        beg = beg - 2 * np.pi
    if end > np.pi:
        end = end - 2 * np.pi

    return ((end - beg) + np.pi) % (2 * np.pi) - np.pi