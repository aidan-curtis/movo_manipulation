from abc import ABC, abstractmethod
from cProfile import label
from email.policy import default
from re import L
from pybullet_planning.pybullet_tools.utils import (LockRenderer, load_pybullet, set_joint_positions, joint_from_name,
                                                    Point, Pose, Euler, draw_pose,
                                                    set_pose, create_box, TAN, get_link_pose,
                                                    get_camera_matrix, get_image_at_pose, tform_point, invert, multiply,
                                                    pixel_from_point, AABB, OOBB, BLUE, RED, YELLOW, link_from_name,
                                                    aabb_contains_point, aabb_from_oobb,
                                                    get_aabb, RGBA, recenter_oobb, get_aabb, draw_oobb,
                                                    aabb_from_points, OOBB, ray_from_pixel,
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

fx = 80
fy = 80
width = 128
height = 128
CAMERA_MATRIX = get_camera_matrix(width, height, fx, fy)
FAR = 3


class Environment(ABC):

    @abstractmethod
    def setup(self):
        pass


    def update_movable_boxes(self, camera_image, ignore_obstacles=[], **kwargs):
        relevant_cloud = [lp for lp in iterate_point_cloud(camera_image, **kwargs)
                          if aabb_contains_point(lp.point, self.room.aabb)
                          ]

        object_points = defaultdict(list)
        for labeled_point in relevant_cloud:
            if (labeled_point.label[0] in self.room.movable_obstacles and
                    labeled_point.label[0] not in ignore_obstacles):
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
                          if aabb_contains_point(lp.point, self.room.aabb)]

        for labeled_point in relevant_cloud:
            if labeled_point.label[0] not in ignore_obstacles:
                if labeled_point.label[1] == -1:
                    point = labeled_point.point
                    self.occupancy_grid.add_point(point)


    def update_visibility(self, camera_pose, camera_image, q):
        surface_aabb = self.visibility_grid.aabb
        camera_pose, camera_matrix = camera_image[-2:]
        grid = self.visibility_grid
        self.gained_vision[q] = set()

        for voxel in grid.voxels_from_aabb(surface_aabb):
            center_world = grid.to_world(grid.center_from_voxel(voxel))
            center_camera = tform_point(invert(camera_pose), center_world)
            distance = center_camera[2]
            pixel = pixel_from_point(CAMERA_MATRIX, center_camera)
            if pixel is not None:
                r, c = pixel
                depth = camera_image.depthPixels[r, c]
                if distance <= depth:
                    self.viewed_voxels.append(voxel)
                    grid.set_free(voxel)
                    self.gained_vision[q].add(voxel)
        return self.gained_vision[q]


    def setup_grids(self):
        self.setup_occupancy_grid()
        self.setup_visibility_grid()
        self.setup_movable_boxes()



    def setup_visibility_grid(self):
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        # surface_aabb = self.room.aabb
        surface_aabb = AABB(lower=self.room.aabb.lower+np.array([-1, -1, 0]),
                            upper=(self.room.aabb.upper[0]+1, self.room.aabb.upper[1]+1, 0.1))
        grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=BLUE
        )
        static_grid = VoxelGrid(
            resolutions, world_from_grid=surface_origin, aabb=surface_aabb, color=BLUE
        )
        for voxel in grid.voxels_from_aabb(surface_aabb):
            grid.set_occupied(voxel)
            static_grid.set_occupied(voxel)

        self.visibility_grid = grid
        self.static_vis_grid = static_grid
        self.gained_vision = dict()



    def setup_default_vision(self, G):
        self.default_vision = dict()
        for i in range(G.t_step):
            q = (0, 0, round(i*2*np.pi/G.t_step, 3))
            self.default_vision[q] = self.gained_vision_from_conf(q)



    def get_optimistic_vision(self, q, G):
        if q in self.gained_vision:
            return self.gained_vision[q]

        self.vis_table = dict()

        default_one = self.default_vision[(0, 0, round(q[2],3))]
        resulting_voxels = set()
        for voxel in default_one:
            voxel_w = np.array(voxel)*np.array(G.res)
            new_voxel_w = voxel_w + np.array([q[0], q[1], 0.1])
            new_voxel = np.rint(np.array(new_voxel_w)/np.array(G.res))
            new_voxel = (new_voxel[0], new_voxel[1], 0)
            if self.static_vis_grid.contains(new_voxel):
                resulting_voxels.add((new_voxel[0], new_voxel[1], 0))

        return resulting_voxels

    def gained_vision_from_conf(self, q):

        grid = self.static_vis_grid
        voxels = set()
        pose = Pose(point=Point(x=q[0], y=q[1], z=0), euler=[0, 0, q[2]])
        camera_pose = multiply(pose, self.camera_pose)

        for voxel in grid.voxels_from_aabb(grid.aabb):
            center_world = grid.to_world(grid.center_from_voxel(voxel))
            center_camera = tform_point(invert(camera_pose), center_world)
            pixel = pixel_from_point(CAMERA_MATRIX, center_camera)
            if pixel is not None:
                voxel_pos = grid.pose_from_voxel(voxel)[0]
                dist = distance(voxel_pos, camera_pose[0])

                if dist < FAR:
                    #self.visibility_grid.set_free(voxel)
                    voxels.add(voxel)
        return voxels


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

        self.camera_pose = get_link_pose(robot_body,
                                         link_from_name(robot_body, "kinect2_rgb_optical_frame"))
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

        # 13 is the link of the optical frame of the rgb camera
        camera_link = link_from_name(self.robot, "kinect2_rgb_optical_frame")
        camera_pose = get_link_pose(self.robot, camera_link)

        camera_image = get_image_at_pose(camera_pose, CAMERA_MATRIX, far=FAR, segment=True)

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

    def get_centered_aabb(self):
        # TODO: Using the base aabb for simplicity. Change later
        centered_aabb, _ = recenter_oobb((get_aabb(self.robot, link=4), Pose()))
        centered_aabb.lower[2] += centered_aabb.upper[2]
        centered_aabb.upper[2] += centered_aabb.upper[2]
        #centered_aabb.lower[1] = centered_aabb.lower[0]
        #centered_aabb.upper[1] = centered_aabb.upper[0]
        return centered_aabb


    def get_centered_oobb(self):
        # TODO: Using the base aabb for simplicity. Change later
        aabb = get_aabb(self.robot, link=4)
        centered_aabb, pose = recenter_oobb((aabb, Pose()))
        return OOBB(centered_aabb, pose)


    def oobb_from_q(self, q):
        oobb = self.centered_oobb
        pose = Pose(point=Point(x=q[0], y=q[1], z=oobb.pose[0][2]), euler=[0,0,q[2]])
        return OOBB(oobb.aabb, pose)

    def aabb_from_q(self, q):
        return aabb_from_oobb(self.oobb_from_q(q))


def distance(vex1, vex2):
    return ((vex1[0] - vex2[0]) ** 2 + (vex1[1] - vex2[1]) ** 2) ** 0.5


def find_min_angle(beg, end):
    if beg > np.pi:
        beg = beg - 2 * np.pi
    if end > np.pi:
        end = end - 2 * np.pi

    return ((end - beg) + np.pi) % (2 * np.pi) - np.pi