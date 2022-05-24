import numpy as np
import open3d as o3d
from pybullet_planning.pybullet_tools.utils import (get_image_aabb, ray_from_pixel, multiply, tform_point,
                                                    Pixel)
from pybullet_planning.pybullet_tools.voxels import MAX_PIXEL_VALUE
from collections import namedtuple

LabeledPoint = namedtuple("LabeledPoint", ["point", "color", "label"])

def get_pointcloud_from_camera_image(camera_image):
    rgb, depth, _, _, _ = camera_image

    # TODO: These intrinsics should be in one location rather than hardcoded in multiple places
    parameters = o3d.camera.PinholeCameraIntrinsic(960, 540, 528.612, 531.854, 477.685, 255.955)
    rgb_image = np.frombuffer(rgb, dtype=np.uint8).reshape(540, 960, -1)
    rgb_image = rgb_image[:,:, [0,1,2]]
    rgb_image = o3d.geometry.Image(np.squeeze(np.array([rgb_image]), axis=0))
    rgb_image = o3d.geometry.Image(np.array(rgb_image))
    depth_image = o3d.geometry.Image(np.array(depth))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_trunc=1000, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                parameters)
    pcd = pcd.voxel_down_sample(voxel_size=0.00003)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    return pcd

def extract_point(camera_image, pixel, world_frame=True):
    # from trimesh.scene import Camera
    rgb_image, depth_image, seg_image, camera_pose, camera_matrix = camera_image
    r, c = pixel
    height, width = depth_image.shape
    assert (0 <= r < height) and (0 <= c < width)
    # body, link = seg_image[r, c, :]
    label = seg_image if seg_image is None else seg_image[r, c]
    ray = ray_from_pixel(camera_matrix, [c, r])  # NOTE: width, height
    depth = depth_image[r, c]
    # assert not np.isnan(depth)
    point_camera = depth * ray

    point_world = tform_point(multiply(camera_pose), point_camera)
    point = (
        point_world if world_frame else point_camera
    )  # TODO: specify frame wrt the robot
    color = rgb_image[r, c, :] / MAX_PIXEL_VALUE
    return LabeledPoint(point, color, label)


def iterate_image(camera_image, step_size=3, aabb=None, **kwargs):
    if aabb is None:
        aabb = get_image_aabb(camera_image.camera_matrix)

    (height, width, _) = camera_image.rgbPixels.shape
    # TODO: clip if out of range
    (x1, y1), (x2, y2) = np.array(aabb).astype(int)
    for r in range(y1, height, step_size):
        for c in range(x1, width, step_size):
            yield Pixel(r, c)


def custom_iterate_point_cloud(
    camera_image, iterator, min_depth=0.0, max_depth=float("inf"), **kwargs
):
    rgb_image, depth_image = camera_image[:2]
    # depth_image = simulate_depth(depth_image)
    for pixel in iterator:

        depth = depth_image[pixel]
        labeled_point = extract_point(camera_image, pixel)
        if (depth <= min_depth) or (depth >= max_depth):
            continue

        yield labeled_point


def iterate_point_cloud(camera_image, **kwargs):
    return custom_iterate_point_cloud(
        camera_image, iterate_image(camera_image, **kwargs), **kwargs
    )
