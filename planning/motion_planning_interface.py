from collections import defaultdict, deque, namedtuple
import pybullet as p
from itertools import combinations, count, cycle, islice, product
import numpy as np


warnings.filterwarnings("ignore")
sys.path.extend(
    [
        "pybullet-planning",
    ]
)

from pybullet_tools.motion_planners.rrt_connect import birrt
from pybullet_tools.separating_axis import separating_axis_theorem
from pybullet_tools.utils import JointInfo, Interval, OOBB,
    Pose, multiply, Point, Euler, get_oobb_vertices, is_circular, wrap_angle, 
    all_between


def check_initial_end(start_conf, end_conf, collision_fn, verbose=True):
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_conf, verbose=verbose):
        print("Warning: initial configuration is in collision")
        return False
    if collision_fn(end_conf, verbose=verbose):
        print("Warning: end configuration is in collision")
        return False
    return True

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
        # TODO: separating axis theorem
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

        return any(flat_collisions)

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


def setup_robot_pybullet(args):
    if args.viewer and args.client == 0:
        client = bc.BulletClient(connection_mode=p.GUI)
        # set_preview(False, client=client)
    else:
        client = bc.BulletClient(connection_mode=p.DIRECT)

    robot_body = load_pybullet(robot_paths[args.robot], fixed_base=True, client=client)
    return robot_body, client



if __name__ == '__main__':
    pass
