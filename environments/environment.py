from abc import ABC, abstractmethod
from pybullet_planning.pybullet_tools.utils import (LockRenderer, load_pybullet, set_joint_positions, joint_from_name, Point, Pose, 
                                                    set_pose, create_box, TAN, get_link_pose,
                                                    get_camera_matrix, get_image_at_pose, tform_point, invert,
                                                    pixel_from_point, AABB, BLUE, RED, link_from_name, aabb_contains_point, 
                                                    get_aabb, RGBA)
from pybullet_planning.pybullet_tools.voxels import (VoxelGrid)
from utils.motion_planning_interface import DEFAULT_JOINTS
from utils.utils import iterate_point_cloud, get_viewcone
import pybullet as p
import os 
import numpy as np
from collections import namedtuple

GRID_HEIGHT = 0.1 # Height of the visibility and occupancy grids
GRID_RESOLUTION = 0.2 # Grid resolutions

LIGHT_GREY = RGBA(0.7, 0.7, 0.7, 1)


Room = namedtuple("Room", ["walls", "floors", "aabb"])

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

    def update_occupancy(self, camera_image, **kwargs):
            relevant_cloud = [ lp for lp in iterate_point_cloud(camera_image, **kwargs)
                if aabb_contains_point(lp.point, self.room.aabb)
            ]
            for labeled_point in relevant_cloud:
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
                    grid.set_free(voxel)
        return grid
       
    def setup_visibility_grid(self):
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        surface_aabb = self.room.aabb
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

    def set_defaults(self, robot):
        joints, values = zip(*[(joint_from_name(robot, k), v) for k, v in DEFAULT_JOINTS.items()])
        set_joint_positions(robot, joints, values)
        
    def setup_robot(self):
        MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
        MOVO_PATH = os.path.abspath(MOVO_URDF)
        robot_body = load_pybullet(MOVO_PATH, fixed_base=True)

        self.set_defaults(robot_body)
        return robot_body

    def plot_grids(self, visibility=False, occupancy=False):
        with LockRenderer():
            p.removeAllUserDebugItems()
            if(visibility):
                self.visibility_grid.draw_intervals()
            if(occupancy):
                self.occupancy_grid.draw_intervals()

    def get_robot_vision(self):
        """
        Gets the rgb and depth image of the robot
        """
        fx = 528.612
        fy = 531.854
        width = 960
        height = 540

        # 13 is the link of the optical frame of the rgb camera
        camera_link = link_from_name(self.robot, "kinect2_rgb_optical_frame")
        camera_pose = get_link_pose(self.robot, camera_link)

        camera_matrix = get_camera_matrix(width, height, fx, fy)
        image_data = get_image_at_pose(camera_pose, camera_matrix)
        viewcone = get_viewcone(camera_matrix=camera_matrix, color=RGBA(1, 1, 0, 0.2))
        set_pose(viewcone, camera_pose)
        camera_image = get_image_at_pose(camera_pose, camera_matrix)

        return camera_pose, camera_image
        

    def create_closed_room(self, length, width, center=[0,0], wall_height=2):

        floor = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor, Pose(Point(x=center[0], y=center[1])))

        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_1, Pose(point=Point(x=center[0], y=center[1]+length/2+wall_thickness/2, z=wall_height/2)))
        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_2, Pose(point=Point(x=center[0], y=center[1]-(length/2+wall_thickness/2), z=wall_height/2)))
        wall_3 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_3, Pose(point=Point(y=center[1], x=center[0]+width/2+wall_thickness/2, z=wall_height/2)))
        wall_4 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=LIGHT_GREY)
        set_pose(wall_4, Pose(point=Point(y=center[1], x=center[0]-(width/2+wall_thickness/2), z=wall_height/2)))
        return Room([wall_1, wall_2, wall_3, wall_4], [floor], get_aabb(floor))


    def create_pillar(self, width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
        return  create_box(w=width, l=length, h=height, color=color, **kwargs)
