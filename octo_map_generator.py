import open3d as o3d
import numpy as np
import os
import sys

import octomap


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])



pcd = o3d.io.read_point_cloud("generated_pointcloud.pcd")

o3d.visualization.draw_geometries([pcd])

min_bounds = pcd.get_min_bound()
max_bounds = pcd.get_max_bound()



ratio = 10
new_size = (int((abs(min_bounds[0]) + abs(max_bounds[0]))*ratio+1), 
			int((abs(min_bounds[0]) + abs(max_bounds[0]))*ratio+1))
print(new_size)








bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bounds[0], -1.02, min_bounds[2])
	, max_bound=(max_bounds[0], 0.1, max_bounds[2]))
pcd = pcd.crop(bb)


cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30,
                                                    std_ratio=5.0)
pcd = pcd.select_by_index(ind)

o3d.visualization.draw_geometries([pcd])


print('octree division')
octree = o3d.geometry.Octree(max_depth=10)
octree.convert_from_point_cloud(pcd, size_expand=0.1)
o3d.visualization.draw_geometries([octree])