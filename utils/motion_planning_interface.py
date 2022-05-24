import pybullet as p
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from utils.grid_generator import robot_to_grid, robot_bb_grid
from pybullet_planning.motion.motion_planners.rrt_connect import birrt
from pybullet_planning.pybullet_tools.utils import OOBB, \
    get_oobb_vertices, is_circular, wrap_angle, \
    all_between, create_box, TAN, load_pybullet, get_aabb, joint_from_name, \
    interval_generator, circular_difference, set_joint_positions, create_box, \
        get_aabb, get_pose
import math

MOVO_URDF = "models/srl/movo_description/movo_robotiq_collision.urdf"
MOVO_PATH = os.path.abspath(MOVO_URDF)

DEFAULT_JOINTS = {
        "pan_joint": -0.07204942405223846,
        "tilt_joint": -0.599216890335083,
        "left_shoulder_pan_joint": -1.193271404355226,
        "left_shoulder_lift_joint": 1.691065746887311,
        "left_arm_half_joint": -2.9308729618144724,
        "left_elbow_joint": -1.3548135629153069,
        "left_wrist_spherical_1_joint": -0.4464001271835176,
        "left_wrist_spherical_2_joint": 1.8807062523185454,
        "left_wrist_3_joint": 1.859177258066345,
        "right_shoulder_pan_joint": -1.990431951339227,
        "right_shoulder_lift_joint": 1.5555300221621264,
        "right_arm_half_joint": -0.3052025219808243,
        "right_elbow_joint": 1.2316691272680973,
        "right_wrist_spherical_1_joint": 0.5568418170632672,
        "right_wrist_spherical_2_joint": -1.9205464764358584,
        "right_wrist_3_joint": -0.059844425487387554,
        "left_gripper_finger1_joint": -0.0008499202079690222,
        "left_gripper_finger2_joint": -0.0,
        "left_gripper_finger3_joint": 0.0,
        "right_gripper_finger1_joint": 0.0,
        "linear_joint": 0.3,
        "right_gripper_finger1_joint": 0,
        "right_gripper_finger2_joint": 0,
        "right_gripper_finger1_inner_knuckle_joint": 0,
        "right_gripper_finger2_inner_knuckle_joint": 0,
        "right_gripper_finger1_finger_tip_joint": 0,
        "right_gripper_finger2_finger_tip_joint": 0,
    }


def check_initial_end(start_conf, end_conf, collision_fn, verbose=True):
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_conf, verbose=verbose):
        print("Warning: initial configuration is in collision")
        return False
    if collision_fn(end_conf, verbose=verbose):
        print("Warning: end configuration is in collision")
        return False
    return True



def create_pillar(width=0.25, length=0.25, height=1e-3, color=None, **kwargs):
    # TODO: use the color when naming
    return  create_box(w=width, l=length, h=height, color=color, **kwargs)
    

def plan_2d_joint_motion(
    robot,
    robot_aabb,
    joints,
    lower_limits,
    upper_limits,
    start_conf,
    end_conf,
    obstacle_oobbs=[],
    attachments=[],
    weights=None,
    resolutions=None,
    algorithm=None,
    disable_collisions=False,
    **kwargs
):
    """Assumed joint indices are x, y, theta"""
    def oobb_flat_vertices(oobb):
        diff_thresh = 0.001
        verts = get_oobb_vertices(oobb)
        verts2d = []
        for vert in verts:
            unique = True
            for vert2d in verts2d:
                if (
                    np.linalg.norm(np.array(vert[:2]) - np.array(vert2d[:2]))
                    < diff_thresh
                ):
                    unique = False
            if unique:
                verts2d.append(vert[:2])
        assert len(verts2d) == 4
        return verts2d

    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)

    circular_joints = [is_circular(robot, joint, **kwargs) for joint in joints]
    lower_limits = [-2*np.pi if circular_joints[li] else ll for li, ll in enumerate(lower_limits)]
    upper_limits = [2*np.pi if circular_joints[li] else ll for li, ll in enumerate(upper_limits)]

    sample_generator = interval_generator(lower_limits, upper_limits, **kwargs)

    def sample_fn():
        return tuple(next(sample_generator))

    def difference_fn(q2, q1, **kwargs):

        return [
            circular_difference(value2, value1) if circular else (value2 - value1)
            for value1, value2, circular in zip(q1, q2, circular_joints)
        ]

    def distance_fn(q1, q2, **kwargs):
        diff = difference_fn(q1, q2)
        return np.linalg.norm(diff, ord=2)

    def refine_fn(q1, q2, num_steps, **kwargs):
        q = q1
        num_steps = num_steps + 1
        for i in range(num_steps):
            positions = (1.0 / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            # q = tuple(positions)
            q = [
                position if not circular_joints[pi] else wrap_angle(position)
                for (pi, position) in enumerate(positions)
            ]
            yield q

    def extend_fn(q1, q2, **kwargs):
        # steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(
            np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=2)
        )
        return refine_fn(q1, q2, steps)


    def limits_fn(q):
        if not all_between(lower_limits, q, upper_limits):
            # print('Joint limits violated')
            # if verbose: print(lower_limits, q, upper_limits)
            return True
        return False


    def collision_fn(q, **kwargs):

        if limits_fn(q):
            return True

        if(disable_collisions):
            return False

        robot_points = robot_to_grid([q[1], 0, q[0]], q[2])

        for point in robot_points:
            if grid_map[point][0] == 1:
                return True

        # TODO: separating axis theorem
        """
        new_oobb = OOBB(
                aabb=robot_aabb,
                pose=Pose(point=Point(x=q[0], y=q[1]), euler=Euler(yaw=q[2])),
            )
        new_oobb_flat = oobb_flat_vertices(new_oobb)
        oobb_flats = [oobb_flat_vertices(o) for o in obstacle_oobbs]

        flat_collisions = []
        for oobb_flat in oobb_flats:
            collision = separating_axis_theorem(new_oobb_flat, oobb_flat)
            flat_collisions.append(collision)
            for attachment_aabb, attachment_grasp in attachments:
                attachment_pose = multiply(new_oobb.pose, attachment_grasp)
                attachment_euler = p.getEulerFromQuaternion(attachment_pose[1])
                attachment_flat = oobb_flat_vertices(OOBB(
                    aabb=attachment_aabb,
                    pose=Pose(point=Point(x=attachment_pose[0][0], y=attachment_pose[0][1]), euler=Euler(yaw=attachment_euler[2])),
                ))
                collision = separating_axis_theorem(new_oobb_flat, attachment_flat)
                # flat_collisions.append(collision)
        """
        return False

    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None

    plan = birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        **kwargs
    )
    return plan


def setup_world(robot, **kwargs):

    floor_size = 6
    floor = create_pillar(width=floor_size, length=floor_size, color=TAN, **kwargs)

        # cracker_box | tomato_soup_can | potted_meat_can | bowl
    side = 1    
    box_mass = 0.2
    height = 1
    
    #box = create_box(
    #            w=side,
    #            l=side,
    #            h=height,
    #            color=RGBA(219 / 256.0, 50 / 256.0, 54 / 256.0, 1.0),
    #            mass=box_mass,
    #            **kwargs
    #        )

    #set_pose(box, Pose(point=Point(x=0, y=0, z=height / 2.0)), **kwargs)


    return []

    

def setup_robot_pybullet():
    p.connect(p.DIRECT)
    robot_body = load_pybullet(MOVO_PATH, fixed_base=True)
    return robot_body

def visualize_path_pcd(pcd, path):
    kin = cph.kinematics.KinematicChain("models/movo_model/movo_robotiq_collision.urdf")
    vis = cph.visualization.Visualizer()
    vis.create_window()


    initial_pose = np.identity(4)
    initial_pose[1:3, 1:3] = [[np.cos(-np.pi/2), -np.sin(-np.pi/2)], [np.sin(-np.pi/2), np.cos(-np.pi/2)]]
    initial_pose[:3, 3] = [0,-1.15,0]


    for i in range(len(path)):
        
        if i%5 == 0 or i == len(path)-1:
            pose = path[i]
            theta = pose[2] + np.pi/2
            new_pose = np.identity(4)
            new_pose[0:2, 0:2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            new_pose[:3, 3] = [-pose[1],pose[0],0]

            new_pose = initial_pose@new_pose

            poses = kin.forward_kinematics(default_joints, new_pose)
            geoms = kin.get_transformed_visual_geometry_map(poses)

            for v in geoms.values():
                vis.add_geometry(v)

    vis.add_geometry(pcd)
    vis.run()



if __name__ == '__main__':
    
    grid_map = plt.imread("grid.png")

    robot_body = setup_robot_pybullet()
    obstacles = setup_world(robot_body)

    joints = []
    values = []
    for elem in DEFAULT_JOINTS:
        joints.append(joint_from_name(robot_body, elem))
        values.append(DEFAULT_JOINTS[elem])

    set_joint_positions(robot_body, joints, values)
    robot_aabb = get_aabb(robot_body)
    

    joints = [joint_from_name(robot_body, "x"),
              joint_from_name(robot_body, "y"),
              joint_from_name(robot_body, "theta")]

    obstacle_oobbs = [OOBB(get_aabb(obj), get_pose(obj)) for obj in obstacles]

    min_vals = [-25, -25, -math.pi*2]
    max_vals = [25, 25, math.pi*2]

    q1 = [0, 0, 0]
    q2 = [6, -1.5, math.pi]


    resolutions = 0.1 * np.ones(len(q2))
    plan = plan_2d_joint_motion(robot_body, robot_aabb, joints, min_vals, max_vals, q1, q2, resolutions=resolutions, obstacle_oobbs=obstacle_oobbs, max_iterations=100)

    print(plan)

    # Visualized the planned path
    for q in plan:

        robot_points = robot_bb_grid([q[1], 0, q[0]], q[2])

        # For visualizing Qs in the grid
        for point in robot_points:
            grid_map[point] = [0.5, 0.5, 0.5, 1]


    cv2.imshow("Planned Path", grid_map)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()

    # Save planned patth
    if os.path.exists("planned_path.txt"):
                os.remove("planned_path.txt")
    for q in plan:
        with open('planned_path.txt', 'a') as f:
            f.write('{} {} {}\n'.format(q[0], q[1], q[2]))
                

    #for plan_step in plan:
    #    set_joint_positions(robot_body, joints, plan_step)
    #    time.sleep(0.1)


    # while(True):
    #     p.setGravity(0,0,0)
    #     time.sleep(0.1)


