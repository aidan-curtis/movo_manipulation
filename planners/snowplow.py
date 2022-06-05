
from planners.rrt import RRT 
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions,
                                                    invert, multiply, pairwise_collisions, Pose, Point, Euler)
import numpy as np
import pybullet as p


class Snowplow(RRT):
    def __init__(self, env):
        super(Snowplow, self).__init__(env)
        self.env = env
        
        # Setup the environment
        self.env.setup()

        self.step_size = [0.05, np.pi/18]
        self.RRT_ITERS = 5000
        self.COLLISION_DISTANCE = 5e-3

        self.joints = [joint_from_name(self.env.robot, "x"),
                       joint_from_name(self.env.robot, "y"),
                       joint_from_name(self.env.robot, "theta")]

    def find_path_obstruction(self, path):
        """
            Find the first movable object bounding box that this path collides with
        """
        for q in path:
            collision = self.env.check_conf_collision(q)
            if(collision):
                return collision

        raise NotImplementedError

    def sample_attachment_base_confs(self, robot, pose, environment=[]):
        obstacles = environment
        base_generator = robot.base_sample_gen(pose)
        for base_conf in base_generator:
            set_joint_positions(self.env.robot, self.joints, base_conf)
            if pairwise_collisions(robot, obstacles, max_distance=self.COLLISION_DISTANCE):
                continue
            yield base_conf


    def sample_attachment(self, movable_box):
        base_conf = next(self.sample_attachment_base_confs(self.env.robot, obj, pose))
        base_pose = Pose(
            point=Point(x=base_conf.positions[0], y=base_conf.positions[1]),
            euler=Euler(yaw=base_conf.positions[2]),
        )
        base_grasp = multiply(invert(base_pose), pose.get_pose())

        return base_conf, base_grasp

    def sample_placement_poses(grasp):
        raise NotImplementedError

    def sample_detachment(self, grasp):

        pose = self.sample_placement_poses(grasp)
        base_pose = multiply(pose.get_pose(), invert(grasp))
        base_point, base_quat = base_pose
        base_euler = p.getEulerFromQuaternion(base_quat)
        base_positions = self.env.robot.get_default_conf()[self.env.robot.base_group]
        base_positions[:3] = [base_point[0], base_point[1], base_euler[2]]
        base_conf = GroupConf(
            robot,
            robot.base_group,
            positions=base_positions,
            **kwargs
        )
        switch = BaseSwitch(obj, parent=WORLD_BODY)

        # TODO: wait for a bit and remove colliding objects
        commands = [switch]
        sequence = Sequence(commands=commands, name="place-{}".format(obj))
        return Tuple(base_conf, sequence)



    def plan_move_path(self, q, movable_box):
        attachment_path = None
        detachment_path = None
        while(attachment_path is None or detachment_path is None):
            attachment_q, attachment_grasp = self.sample_attachment(movable_box)
            detach_q = self.sample_detachment(attachment_grasp)
            self.get_path(q, attachment_q)
            self.get_path(attachment_q, detach_q)

        return attachment_path, detachment_path, attachment_grasp


    def get_plan(self):
        
        camera_pose, image_data = self.env.get_robot_vision()
        self.env.update_visibility(camera_pose, image_data)
        self.env.update_occupancy(image_data)

        self.env.plot_grids(visibility=False, occupancy=True, movable=True)
        current_q, complete = self.env.start, False

        while(not complete):
            final_path = self.get_path(current_q, self.env.goal)
            if(final_path is None):
                print("No direct path to goal")
                relaxed_final_path = self.get_path(current_q, self.env.goal, ignore_movable=True)
                if(relaxed_final_path is None):
                    print("No indirect path to goal :(")
                else:
                    obstruction = self.find_path_obstruction(relaxed_final_path)
                    print("Found path through obstacle: "+str(obstruction))
                    attach_path, detach_path, grasp = self.plan_move_path()
                    current_q, complete = self.execute_path(final_path, attach_path)
                    assert complete
                    current_q, complete = self.execute_path(final_path, detach_path, attachments=[grasp])
                    assert complete
            else:
                current_q, complete = self.execute_path(final_path)

        wait_if_gui()
    
