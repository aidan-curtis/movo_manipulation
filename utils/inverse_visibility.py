from pybullet_planning.pybullet_tools.utils import BodySaver, get_link_pose, get_image_at_pose, get_pose, \
    invert, multiply, get_link_name, pixel_from_ray, pairwise_collisions
import random
MAX_KINECT_DISTANCE = 2.5
COLLISION_DISTANCE = 5e-3  # Distance from fixed obstacles

class Camera(object):  # TODO: extend Object?
    def __init__(
        self,
        robot,
        link,
        optical_frame,
        camera_matrix,
        max_depth=MAX_KINECT_DISTANCE,
        client=None,
        **kwargs
    ):
        self.robot = robot
        self.client = client
        self.link = link  # TODO: no longer need this
        self.optical_frame = optical_frame
        self.camera_matrix = camera_matrix
        self.max_depth = max_depth
        self.kwargs = dict(kwargs)

    def get_pose(self):
        return get_link_pose(self.robot, self.optical_frame, client=self.client)

    def get_image(self, segment=True, segment_links=False, **kwargs):
        # TODO: apply maximum depth
        # TODO: noise model
        return get_image_at_pose(
            self.get_pose(),
            self.camera_matrix,
            tiny=False,
            segment=segment,
            segment_links=segment_links,
            client=self.client,
        )  # TODO: OpenCV


    def object_visible(self, obj):
        camera_matrix = self.camera_matrix
        obj_pose = get_pose(obj, client=self.client)
        ray = multiply(invert(self.robot.cameras[0].get_pose()), obj_pose)[0]
        image_pixel = pixel_from_ray(camera_matrix, ray)
        width, height = dimensions_from_camera_matrix(self.camera_matrix)
        if(image_pixel[0]<width and image_pixel[0]>=0 and image_pixel[1]<height and image_pixel[1]>=0 and ray[2]>0):
            return True
        return False

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, get_link_name(self.robot.body, self.optical_frame)
        )

def sample_visibility_base_confs(
    robot, obj, pose, environment=[], **kwargs
):
    robot_saver = BodySaver(robot, **kwargs)  # TODO: reset the rest conf
    obstacles = environment

    base_generator = uniform_pose_generator(
        robot, pose.get_pose(), reachable_range=(0.5, 1.0)
    )
    # base_generator = learned_pose_generator(robot, grasp_pose, arm=side, grasp_type='top') # TODO: top & side
    for base_conf in base_generator:
        robot_saver.restore()
        pose.assign()
        base_conf = GroupConf(robot, robot.base_group, positions=base_conf, **kwargs)
        base_conf.assign()
        # TODO: check base limits
        if pairwise_collisions(robot, obstacles, max_distance=COLLISION_DISTANCE):
            continue
        yield base_conf


def get_plan_mobile_look_fn(robot, camera, environment=[], max_head_attempts=10, max_base_attempts=100, **kwargs):
    robot_saver = BodySaver(robot, client=robot.client)
    def fn(obj, pose):
        while(True):
            robot_saver.restore()
            # TODO: Return a head conf that will lead to visibility of obj at pose
            if(robot.head_group is None):
                return None
            else:
                pose.assign()
                limits = list(robot.get_group_limits(robot.head_group))
                num_base_attempts = 0
                for base_conf in sample_visibility_base_confs( robot, obj, pose, **kwargs ):
                    visible = False
                    base_conf.assign()
                    num_head_attempts = 0
                    while(not visible):
                        random_head_pos = [random.uniform(*limit) for limit in zip(*limits)]
                        robot.set_group_positions(robot.head_group, random_head_pos)
                        visible = robot.cameras[0].object_visible(obj)
                        num_head_attempts+=1
                        if(num_head_attempts>=max_head_attempts):
                            break
                    if(num_head_attempts>=max_head_attempts):
                        continue
                    gp = random_head_pos
                    current_hq = GroupConf(robot, robot.head_group, gp, client=robot.client)
                    num_base_attempts += 1 

 
                    yield (base_conf, current_hq)
                    if(num_base_attempts>max_base_attempts):
                        return None
            num_attempts+=1
    return fn


if __name__ == "__main__":
    raise NotImplementedError