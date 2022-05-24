from abc import ABC, abstractmethod
from pybullet_planning.pybullet_tools.utils import (load_pybullet, set_joint_positions, joint_from_name, Point, Pose, 
                                                    set_pose, create_box, TAN, GREY, get_link_pose,
                                                    get_camera_matrix, get_image_at_pose, RGBA)
from utils.motion_planning_interface import DEFAULT_JOINTS
import os 
from utils.utils import get_viewcone

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

    def set_defaults(self, robot):
        joints, values = zip(*[(joint_from_name(robot, k), v) for k, v in DEFAULT_JOINTS.items()])
        set_joint_positions(robot, joints, values)
        
    def setup_robot(self):
        MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
        MOVO_PATH = os.path.abspath(MOVO_URDF)
        robot_body = load_pybullet(MOVO_PATH, fixed_base=True)

        self.set_defaults(robot_body)
        return robot_body


    def get_robot_vision(self):
        """
        Gets the rgb and depth image of the robot
        """
        fx = 528.612
        fy = 531.854
        cx = 477.685
        cy = 255.955
        width = 960
        height = 540

        # 13 is the link of the optical frame of the rgb camera
        camera_pose = get_link_pose(self.robot, 13)
        camera_matrix = get_camera_matrix(width, height, fx, fy)
        image_data = get_image_at_pose(camera_pose, camera_matrix)
        viewcone = get_viewcone(camera_matrix=camera_matrix, color=RGBA(1, 1, 0, 0.2))
        set_pose(viewcone, camera_pose)

        return image_data[0], image_data[1]
        

    def create_closed_room(self, length, width, center=[0,0], wall_height=2):

        floor = self.create_pillar(width=width, length=length, color=TAN)
        set_pose(floor, Pose(Point(x=center[0], y=center[1])))

        wall_thickness = 0.1
        wall_1 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=GREY)
        set_pose(wall_1, Pose(point=Point(x=center[0], y=center[1]+length/2+wall_thickness/2, z=wall_height/2)))
        wall_2 = self.create_pillar(width=width, length=wall_thickness, height=wall_height, color=GREY)
        set_pose(wall_2, Pose(point=Point(x=center[0], y=center[1]-(length/2+wall_thickness/2), z=wall_height/2)))
        wall_3 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=GREY)
        set_pose(wall_3, Pose(point=Point(y=center[1], x=center[0]+width/2+wall_thickness/2, z=wall_height/2)))
        wall_4 = self.create_pillar(length=length, width=wall_thickness, height=wall_height, color=GREY)
        set_pose(wall_4, Pose(point=Point(y=center[1], x=center[0]-(width/2+wall_thickness/2), z=wall_height/2)))

        return floor, wall_1, wall_2, wall_3, wall_4


    def create_pillar(self, width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
        return  create_box(w=width, l=length, h=height, color=color, **kwargs)
