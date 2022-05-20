import numpy as np
import open3d as o3d


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