import numpy as np
import open3d as o3d

from pybullet_planning.pybullet_tools.utils import (dimensions_from_camera_matrix, ray_from_pixel, 
                                                create_mesh, Pose, link_from_name,
                                                get_all_links, tform_point, add_line)


def get_pointcloud_from_rgb_and_depth(rgb, depth):
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


def cone_vertices_from_base(base):
    return [np.zeros(3)] + base


def cone_mesh_from_support(support):
    assert len(support) == 4
    vertices = cone_vertices_from_base(support)
    faces = [(1, 4, 3), (1, 3, 2)]
    for i in range(len(support)):
        index1 = 1 + i
        index2 = 1 + (i + 1) % len(support)
        faces.append((0, index1, index2))
    return vertices, faces


def get_viewcone_base(depth=5, camera_matrix=None):
    #if camera_matrix is None:
    #    camera_matrix = PR2_CAMERA_MATRIX
    width, height = dimensions_from_camera_matrix(camera_matrix)
    vertices = []
    for pixel in [(0, 0), (width, 0), (width, height), (0, height)]:
        ray = depth * ray_from_pixel(camera_matrix, pixel)
        vertices.append(ray[:3])
    return vertices


def get_viewcone(depth=5, camera_matrix=None, **kwargs):
    mesh = cone_mesh_from_support(
        get_viewcone_base(depth=depth, camera_matrix=camera_matrix)
    )
    assert mesh is not None
    return create_mesh(mesh, **kwargs)