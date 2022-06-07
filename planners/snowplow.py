from planners.rrt import RRT
from pybullet_planning.pybullet_tools.utils import (wait_if_gui, joint_from_name, set_joint_positions,
                                                    invert, multiply, pairwise_collisions, Pose, Point, Euler,
                                                    point_from_pose, sample_directed_reachable_base, get_pose,
                                                    get_aabb_center)
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
            if collision:
                return collision

    def directed_pose_generator(self, point, **kwargs):
        while True:
            base_values = sample_directed_reachable_base(self, point, **kwargs)
            if base_values is None:
                break
            yield tuple(list(base_values)+[0.1])


    def base_sample_gen(self, point):
        return self.directed_pose_generator(point, reachable_range=(0.8, 0.8))


    def sample_attachment_base_confs(self, robot, movable_box):
        base_generator = self.base_sample_gen(get_aabb_center(movable_box.aabb))
        for base_conf in base_generator:
            base_conf = base_conf[0:3]
            set_joint_positions(self.env.robot, self.joints, base_conf)
            if self.env.check_conf_collision(base_conf):
                continue
            return base_conf


    def sample_attachment(self, movable_box):
        base_conf = self.sample_attachment_base_confs(self.env.robot, movable_box)
        base_pose = Pose(
            point=Point(x=base_conf[0], y=base_conf[1]),
            euler=Euler(yaw=base_conf[2]),
        )
        #base_grasp = multiply(invert(base_pose), pose.get_pose())
        #print(base_grasp)

        return base_conf #, base_grasp

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



    def plan_move_path(self, movable_box, q):
        attachment_path = None
        detachment_path = None
        while(attachment_path is None): #or detachment_path is None):
            attachment_q = self.sample_attachment(movable_box)
            #detach_q = self.sample_detachment(attachment_grasp)
            attachment_path = self.get_path(q, attachment_q)
            #self.get_path(attachment_q, detach_q)
        print(attachment_path)
        wait_if_gui()
        return attachment_path#, detachment_path, attachment_grasp


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
                    attach_path, detach_path, grasp = self.plan_move_path(obstruction, current_q)
                    current_q, complete = self.execute_path(final_path, attach_path)
                    assert complete
                    current_q, complete = self.execute_path(final_path, detach_path, attachments=[grasp])
                    assert complete
            else:
                current_q, complete = self.execute_path(final_path)

        wait_if_gui()
    
